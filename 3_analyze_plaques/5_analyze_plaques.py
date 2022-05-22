"""
Step 3. Analyze each cell of the grid to search for lysis plaque.
For each plate, the grid was previously located. 
"""

from ast import keyword
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd
import shutil

import cv2
from cv2 import createBackgroundSubtractorMOG2
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
from skimage.measure import find_contours
from skimage.restoration import rolling_ball
from scipy.ndimage import binary_fill_holes
from scipy.stats import mode

import napari

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_farid"
table_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_grids.npy"
save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/4_predictions.csv"

grids = np.load(table_path)  #TODO: un peu sale car implique que l'ordre dans les grilles == ordre de chargement des images

def pad_image(im, target_shape=(1500, 1500)):
    """
    In Napari, all images must have the same shape. This function pads the images at the right and at the bottom
    (to avoid modifying points coordinates in the image) to get all images with a final shape of target_shape.
    """
    dx, dy = target_shape[0] - im.shape[0], target_shape[1] - im.shape[1]
    if dx > 0 and dy > 0: 
        pad_x = np.zeros((dx, im.shape[1]))
        pad_y = np.zeros((target_shape[0], dy))
        im_with_defined_shape = np.concatenate([im, pad_x], axis=0)
        im_with_defined_shape = np.concatenate([im_with_defined_shape, pad_y], axis=1)
        return im_with_defined_shape.astype("uint8")
    elif dx > 0:
        pad_x = np.zeros((dx, im.shape[1]))
        im_with_defined_shape = np.concatenate([im, pad_x], axis=0)
        return im_with_defined_shape[:, :target_shape[1]].astype("uint8")
    elif dy > 0:
        pad_y = np.zeros((im.shape[0], dy))
        im_with_defined_shape = np.concatenate([im, pad_y], axis=1)
        return im_with_defined_shape[:target_shape[0], :].astype("uint8")
    else:
        return im[:target_shape[0], :target_shape[1]].astype("uint8")

def find_plaque(patch):
    p = 0
    contours = find_contours(patch)
    for c in contours:
        perimeter = cv2.arcLength(c.astype("float32"), True)
        area = cv2.contourArea(c.astype("float32"))
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        print(circularity)
        if area > 2000 and circularity > 0.8:
            p = 1
    return p

predictions = []
for k, ifl in enumerate(sorted(os.listdir(directory))[:5]):
    extension = ifl.split(".")[-1]  # get the file extension

    # if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
    if ifl == "spotassay_019_01.jpg":
        if k % 100 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")

        print(ifl)

        im = imageio.imread(directory + "/" + ifl)
        grid = grids[k]

        height, width = im.shape
        # background = rolling_ball(im)

        patch_x, patch_y = 50, 50  # rayon du patch
        pred = []
        for j, coord in enumerate(grid):
            x, y = coord
            top, bottom, left, right = np.max([0, x - patch_x]), np.min([x + patch_x, height]), np.max([0, y - patch_y]), np.min([width, y + patch_y])
            patch = im[top:bottom, left:right]# - background[top:bottom, left:right]
            thresh = threshold_otsu(patch)
            patch = binary_fill_holes(patch > thresh)

            p = find_plaque(patch)
            pred.append(p)

            if p == 1:
                print(j)
                plt.imshow(patch)
                plt.show()
        
        d = {"image": ifl}
        for j, p in enumerate(pred):
            d[str(j)] = p
        predictions.append(d)

print(d)
predictions = pd.DataFrame(d, index=[0])

print(predictions)
