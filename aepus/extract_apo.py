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

from skimage import exposure
from skimage.filters.rank import mean_bilateral
from skimage.transform import radon, rescale
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import scipy
from numpy import unravel_index

def get_enhanced_imaged(img_arr):
    
    img_adapt_hist = exposure.equalize_adapthist(img_arr, clip_limit=0.01)
    return img_adapt_hist

def get_radon_transform(image, n_angles=None):
    
    if n_angles is None:  
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)  
    else:
        theta = np.linspace(0., 180., n_angles, endpoint=False)
    sinogram = radon(image, circle=False, theta=theta)
    
    return sinogram, theta

def get_aponeurosis(img_arr, sigma=[4, 4], 
                    sup_apo_max_depth=0.3, 
                    sup_apo_max_angle=45,
                    deep_apo_min_depth=0.5,
                    deep_apo_max_angle=60,
                    extended_output=False):
    
    # Enhance the image
    img_enhanced = get_enhanced_imaged(img_arr)
    
    # Calculate Radon transform
    # Resolution 0.3 C
    sinogram, theta = get_radon_transform(img_enhanced)
    
    # Calaculate baseline for Radon transform
    sinogram_def, theta = get_radon_transform(np.ones(img_enhanced.shape)*img_enhanced.mean())
    
    # Subtract the baseline and smooth the sinogram with 2D gaussian
    sinogram_img_filt = scipy.ndimage.filters.gaussian_filter(sinogram - sinogram_def, sigma, mode='constant')
    
    # Get indicies of the central of y axis
    y_center_idx = int(sinogram.shape[0]/2)
    
    # Search the aponeurosis in the predefined areas
    # Search for superficial aponeurosis
    sup_apo_min_dist_y_from_center=int((1/2 - sup_apo_max_depth)*img_arr.shape[0])
    sup_apo_angle_mask_x = np.logical_and(theta>=90 - sup_apo_max_angle, theta<=90 + sup_apo_max_angle)
    
    # Select a part of the radon transform relative to the center of the sinogram
    temp_arr = sinogram_img_filt[y_center_idx + sup_apo_min_dist_y_from_center:, sup_apo_angle_mask_x]
    
    # Find maximum
    sup_apo_max_idx = np.array(np.unravel_index(temp_arr.argmax(), temp_arr.shape))
    # Convert to the global radon indices of the original image
    sup_apo_max_idx += (int(img_arr.shape[0]/2) + sup_apo_min_dist_y_from_center, np.where(sup_apo_angle_mask_x)[0][0])
   
    # Search for deep aponeurosis
    deep_apo_min_dist_y_from_center=int((deep_apo_min_depth - 1/2)*img_arr.shape[0])
    deep_apo_angle_mask_x = np.logical_and(theta>=90 - deep_apo_max_angle, theta<=90 + deep_apo_max_angle)
    
    temp_arr = sinogram_img_filt[:y_center_idx - deep_apo_min_dist_y_from_center, deep_apo_angle_mask_x]

    # Find maximum
    deep_apo_max_idx = np.array(np.unravel_index(temp_arr.argmax(), temp_arr.shape))
    # Convert to the global radon indices of the original image
    deep_apo_max_idx += (int(img_arr.shape[0]/2) - y_center_idx, np.where(deep_apo_angle_mask_x)[0][0])
    
    ## Translate the second component of tuples into angle. Correct the angle to fit array coordinates
    sup_apo_max_idx[1] = theta[sup_apo_max_idx[1]] + 90
    deep_apo_max_idx[1] = theta[deep_apo_max_idx[1]] + 90
    
    if not extended_output:
        return sup_apo_max_idx, deep_apo_max_idx
    else: 
        return sup_apo_max_idx, deep_apo_max_idx, img_enhanced, sinogram, sinogram_def, sinogram_img_filt, theta