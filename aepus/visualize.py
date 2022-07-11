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

from PIL import Image, ImageOps
import numpy as np
import cv2

def annotate_image(img, fascicle_angle, sup_apo_coords, deep_apo_coords, line_thickness=5):

    # coods = (coord_x, angle_degree)
    def get_line_points(coords, img_shape):
        
        # Change the coordinates from image system (origin in the center, y - up, x - right)
        # To array coordinates (origin moved to the left top corner, y is flipped)
        slope = -np.tan(np.deg2rad(coords[1]))
        intercept = img_shape[0] - (coords[0] - np.tan(np.deg2rad(coords[1])) * img_shape[1]/2)
        
        # Take x as a reference
        x_vals = np.array([0, img_shape[1]])
        y_vals = intercept + slope * x_vals
        
        if y_vals[0] < 0:
            # Take y = 0 as a reference
            y_vals[0] = 0
            x_vals[0] = (y_vals[0] - intercept)/slope
                
        if y_vals[0] > img_shape[0]:
            # Take y = img_shape[0] as a reference
            y_vals[0] = img_shape[0]
            x_vals[0] = (y_vals[0] - intercept)/slope
            
        if y_vals[1] < 0:
            # Take y = 0 as a reference
            y_vals[1] = 0
            x_vals[1] = (y_vals[1] - intercept)/slope
                
        if y_vals[1] > img_shape[0]:
            # Take y = img_shape[0] as a reference
            y_vals[1] = img_shape[0]
            x_vals[1] = (y_vals[1] - intercept)/slope
            
        x_vals = x_vals.astype(np.int)
        y_vals = y_vals.astype(np.int)
            
        return (x_vals[0], y_vals[0]), (x_vals[1], y_vals[1])
            
    # Convert the image to RGB
    img = (img.astype(np.float32))/img.max()
    img_color = cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    
    # Plot sup. aponeurosis
    points = get_line_points(sup_apo_coords, img.shape)

    image = cv2.line(img_color, points[0], points[1], (255, 0, 0), line_thickness)

    # Plot deep. aponeurosis
    points = get_line_points(deep_apo_coords, img.shape)
    
    image = cv2.line(img_color, points[0], points[1], (0, 0, 255), line_thickness)
    
    # Plot fascicles aponeurosis
    points = get_line_points(((sup_apo_coords[0] + deep_apo_coords[0])/2, fascicle_angle), img.shape)
    
    image = cv2.line(img_color, (points[0]), points[1], (0, 255, 0), line_thickness)

    return image