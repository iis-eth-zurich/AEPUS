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

from AEPUS.aepus.extract_apo import get_enhanced_imaged, get_radon_transform, get_aponeurosis
from AEPUS.aepus.extract_fasc import get_fasc_area_mask, get_fasc_area_cropped
from AEPUS.aepus.extract_fasc import get_fascicle_inclination
import numpy as np

def prepare_raw_img(image_data, db_range=80):
    
    image = 20 * np.log10(image_data.copy()/image_data.max())
    image[image < (-db_range)] = -db_range
    image += db_range
    image = image/image.max()
    
    return image

def process_image(img, db_range=80, raw_image=False):

    results = {}

    if raw_image:
        img = prepare_raw_img(img, db_range=db_range)

    # Find aponeurosis
    sup_apo, deep_apo = get_aponeurosis(img)
    results['sup_apo_coords'] = sup_apo
    results['deep_apo_coords'] = deep_apo
    
    # Create the mask for the fascicles
    fasc_mask, top_line, bot_line = get_fasc_area_mask(img, sup_apo, deep_apo, flip=True)
    
    results['fasc_mask'] = fasc_mask

    # Find inclination angle of the fascicles
    weighted_angle, metric, theta, _, submasks, sinograms, weights = get_fascicle_inclination(img, 
                                                                                              fasc_mask,
                                                                                              top_line,
                                                                                              bot_line)


    results['fasc_angle'] = weighted_angle
    results['fasc_metric'] = metric
    results['theta'] = theta
    results['submasks'] = submasks
    results['sinograms'] = sinograms
    results['weights'] = weights
    
    return results