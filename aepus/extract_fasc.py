"""
   Copyright (C) 2022 ETH Zurich. All rights reserved.

   Author: Sergei Vostrikov, ETH Zurich

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
import numpy as np
import scipy
import math

from AEPUS.aepus.extract_apo import get_radon_transform, get_enhanced_imaged
from AEPUS.aepus.inscribe_circles import get_circles

# Create a mask for the
def get_fasc_area_mask(img_arr, sup_apo_coords, deep_apo_coords, extra_cut_width=0.05, flip=True):
    
    
    extra_cut_width = int(img_arr.shape[1] * extra_cut_width)
    
    # All calculations are in array coord system
    # Angle does not changes but intercept should be recalculated
    sup_apo_slope = np.tan(np.deg2rad(sup_apo_coords[1]))
    sup_apo_intercept = int(sup_apo_coords[0] - sup_apo_slope*img_arr.shape[1]/2) - extra_cut_width

    deep_apo_slope = np.tan(np.deg2rad(deep_apo_coords[1]))
    deep_apo_intercept = int(deep_apo_coords[0] - deep_apo_slope*img_arr.shape[1]/2) + extra_cut_width
    
    # Preapre the mesh
    x = np.arange(0, img_arr.shape[1])
    y = np.arange(0, img_arr.shape[0])

    X, Y = np.meshgrid(x, y)
    
    # Calc values of superf and deep apon. alonge the x
    sup_apo_vals = sup_apo_slope*x + sup_apo_intercept
    deep_apo_vals = deep_apo_slope*x + deep_apo_intercept
    
    
    mask = np.logical_and(Y > deep_apo_vals, Y < sup_apo_vals)
    
    

    if flip is True:
        
        # Flip the sup apo and deep apo lines
        sup_apo_slope = -sup_apo_slope
        sup_apo_intercept = img_arr.shape[0] - sup_apo_intercept
        
        deep_apo_slope = -deep_apo_slope
        deep_apo_intercept = img_arr.shape[0] - deep_apo_intercept
        
        
        # Return flipped mask (relative to y axis) which will correspond to the img_array
        return np.flip(mask, axis=0), [sup_apo_intercept, sup_apo_slope], [deep_apo_intercept, deep_apo_slope]
    
    else:
        # Return the mask which will correspond to classical axes y-> top x-> right
        return mask, [sup_apo_intercept, sup_apo_slope], [deep_apo_intercept, deep_apo_slope]

# Crop the area around the mask + cut of side zones
def get_fasc_area_cropped(mask, img, x_cut_width=0.02):
    
    temp = np.nonzero(mask[:,0])
    y_min_1 = min(temp[0])
    y_max_1 = max(temp[0])
    temp = np.nonzero(mask[:,-1])
    y_min_2 = min(temp[0])
    y_max_2 = min(temp[0])

    y_min = min(y_min_1, y_min_2)
    y_max = max(y_max_1, y_max_2)
    
    x_cut_px = int(x_cut_width*img.shape[1])
    
    if x_cut_px == 0:
        return (img)[y_min:y_max, :], mask[y_min:y_max, :]      
    else:
        return (img)[y_min:y_max, x_cut_px: -x_cut_px], mask[y_min:y_max, x_cut_px:-x_cut_px]

# Crop the area around the mask
def get_mask_area_cropped(mask, img):
    
    temp = np.nonzero(mask)
    y_min = temp[0].min()
    y_max = temp[0].max()
    
    x_min = temp[1].min()
    x_max = temp[1].max()
    
    return mask[y_min:y_max, x_min:x_max], img[y_min:y_max, x_min:x_max]

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

# img_fascicle image with only fascicles
def get_fascicle_inclination(img_fascicle, 
                             img_fascicle_mask, 
                             top_line, 
                             bot_line, 
                             n_circles=6,
                             x_cut_width=0.015,
                             sigma_subimages=10):
    
    # Enhance the image
    img_enhanced = get_enhanced_imaged(img_fascicle)
    
    # Identify the circlular regions
    img_shape = img_fascicle.shape
    
    x_circ, y_circ, rad_circ = get_circles(img_fascicle_mask, 
                                           top_line, 
                                           bot_line, 
                                           n_circles=n_circles, 
                                           x_cut_width=x_cut_width)
    
    # Generate the submasks and subimages
    subimages = []
    submasks = []
    
    for i in range(n_circles):
        # Create a circular mask
        mask_temp = create_circular_mask(img_shape[0], 
                                        img_shape[1], 
                                        center=(math.floor(x_circ[i]), math.floor(y_circ[i])), 
                                        radius=math.floor(rad_circ[i]))
                                    
        # Crop the mask and image
        mask_cropped, img_cropped = get_mask_area_cropped(mask_temp, img_enhanced)
        
        # Add them to the list
        subimages.append(img_cropped)
        submasks.append(mask_cropped)
        
    # Calculate radon transform
    sinograms = []
    
    
    # n_angles
    n_angles = math.floor(2*rad_circ.max())
    
    for i in range(n_circles):
        # get radon transform of the image
        sinogram, theta = get_radon_transform(subimages[i]*submasks[i], n_angles=n_angles)
        
        # get default radon
        # blur subimage 
        mean_subimage = gaussian_filter(subimages[i].copy(), sigma=sigma_subimages) * submasks[i]
        sinogram_def, theta = get_radon_transform(mean_subimage,  n_angles=n_angles)
        
        # Compensate for the base shape
        sinogram = sinogram - sinogram_def
                                             
        sinograms.append(sinogram.astype(np.float32))
        
        
    # Calculate wiggliness metric 
    
    # Energy
    metric = np.zeros((n_circles, n_angles))
    for i in range(n_circles):
        metric[i] = np.sum(sinograms[i]**2, axis=0)
        
    # Weight the std curves for different regions according to the square of the circular region
    # and sum them up
    # Then find the maximum
    weights = rad_circ**2/(rad_circ**2).sum()
    metric_weighted = (metric * weights.reshape(-1,1)).sum(axis=0)

    # Fit Gaussian with intercept to find the center of the peak
    # Uses assumption of symmetrical peak
    # Function to be fitted
    def gauss(x, x0, y0, ybase, sigma):
        p = [x0, y0, ybase, sigma]
        return p[1]* np.exp(-((x-p[0])/p[3])**2) + ybase

    # Initialization parameters
    p0 = [theta[metric_weighted.argmax()], metric_weighted.max(), metric_weighted.max(), 20]
    # Fit the data with the function
    fit, tmp = curve_fit(gauss, 
                         theta, 
                         metric_weighted, 
                         p0=p0)

    weighted_angle = fit[0]

    return weighted_angle + 90, metric, theta, subimages, submasks, sinograms, weights