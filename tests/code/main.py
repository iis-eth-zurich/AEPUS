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

from os.path import abspath
from os.path import dirname as up
import sys
import numpy as np
from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt
import time

# Insert path to the library to system path
path_to_lib = up(up(up(up(abspath(__file__)))))
sys.path.insert(0, path_to_lib)

from AEPUS.aepus.extract_features import process_image
from AEPUS.aepus.visualize import annotate_image

if __name__ == '__main__':

    print("--- Starting feature extraction from the Image 1 ---")

    # Load Image 1
    im = Image.open(path_to_lib + '/AEPUS/tests/data/image_0.tif')
    image = np.array(ImageOps.grayscale(im))

    start_time = time.time()
    result = process_image(image)
    print("--- Execution time: %s seconds    ---" % (time.time() - start_time))

    # Annotate and visualize the image
    img_annotated = annotate_image(image,
                                   result['fasc_angle'],
                                   result['sup_apo_coords'],
                                   result['deep_apo_coords'])

    plt.imshow(img_annotated)
    plt.title("Sample image 1")
    plt.xlabel("X [pixels]")
    plt.ylabel("Z [pixels]")

    plt.savefig('sample_image_1.png', bbox_inches='tight')

    print()
    print("--- Starting feature extraction from the Image 2 ---")

    # Load Image 2
    im = Image.open(path_to_lib + '/AEPUS/tests/data/image_1.tif')
    image = np.array(ImageOps.grayscale(im))

    start_time = time.time()
    result = process_image(image)
    print("--- Execution time: %s seconds    ---" % (time.time() - start_time))

    # Annotate and visualize the image
    img_annotated = annotate_image(image,
                                result['fasc_angle'],
                                result['sup_apo_coords'],
                                result['deep_apo_coords'])

    plt.imshow(img_annotated)
    plt.title("Sample image 2")
    plt.xlabel("X [pixels]")
    plt.ylabel("Z [pixels]")

    plt.savefig('sample_image_2.png', bbox_inches='tight')

    print()
    print("--- Successfully finished                        ---")
    print("--- Please, explore the files                    ---")
    print("--- sample_image_1.png and sample_image_2.png    ---")
    print("--- in the script folder.                        ---")