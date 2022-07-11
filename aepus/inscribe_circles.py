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

import numpy as np
import sys
import math

def get_median_line(line_1, line_2):
    
    # line = [intercept, slope]
    
    # Find intersection point
    # If lines are not parallel
    if abs(line_1[1] - line_2[1]) > sys.float_info.epsilon:
        x = (line_2[0] - line_1[0]) / (line_1[1] - line_2[1])
        y = line_1[1] * x + line_1[0]

        # Calculate slope of the median line
        slope = np.tan((np.arctan(line_1[1]) + np.arctan(line_2[1]))/2)
        intercept = y - slope*x
    else:
        slope = line_1[1]
        intercept = (line_1[0] + line_2[0])/2
    
    a = slope
    b = -1
    c = intercept
    
    return a, b, c

def calc_dist_point_to_line(x, y, a, b, c):
    
    return np.abs(a*x + b*y + c)/np.sqrt(a*a + b*b)
    
    
def get_circles(img_mask, top_line, bot_line, n_circles=1, x_cut_width=0.02):
    
    # Reconstruct lines (get coeffs of line equation)
    a_top = top_line[1] # slope
    b_top = -1
    c_top = top_line[0] # intercept
    
    a_bot = bot_line[1] # slope
    b_bot = -1
    c_bot = bot_line[0] # intercept
    
    # get coeff of the median line 
    a_med, b_med, c_med = get_median_line(top_line, bot_line)
    
    # Get points of the median line
    
    x = np.arange(img_mask.shape[1])
    y = a_med*x + c_med
    
    # Calculate the distance from median line points to top_line, y=0 and y = img_mask.shape[1]
    
    dist_to_top = calc_dist_point_to_line(x, y, a_top, b_top, c_top)
    dist_to_left = calc_dist_point_to_line(x, y, 1, 0, -0)
    dist_to_right = calc_dist_point_to_line(x, y, 1, 0, -img_mask.shape[1])
    
    # Account for x_cut_width. (we don't want the circles to be affected by border poor quality image areas)
    x_cut_px = int(x_cut_width*img_mask.shape[1])
    dist_to_left = dist_to_left - x_cut_px
    dist_to_right = dist_to_right - x_cut_px
    
    # Calc the coordinates + radius for the left circle
    idx = np.argmin(np.abs(dist_to_top - dist_to_left))
    x_left = math.floor(x[idx])
    y_left = math.floor(y[idx])
    rad_left = dist_to_top[idx]
    
    # Calc the coordinates + radius for the right circle
    idx = np.argmin(np.abs(dist_to_top - dist_to_right))
    x_right = math.floor(x[idx])
    y_right = math.floor(y[idx])
    rad_right = dist_to_top[idx]
    
    
    # Make n_circles
    if n_circles == 1:

        x_circles = np.linspace(x_left, x_right, 3)
        y_circles = np.linspace(y_left, y_right, 3)
        rad_circles = np.linspace(rad_left, rad_right, 3)

        # take only central
        x_circles = np.array(x_circles[1])
        y_circles = np.array(y_circles[1])
        rad_circles = np.array(rad_circles[1])

    else:
        x_circles = np.linspace(x_left, x_right, n_circles)
        y_circles = np.linspace(y_left, y_right, n_circles)
        rad_circles = np.linspace(rad_left, rad_right, n_circles)
    
    return x_circles, y_circles, rad_circles