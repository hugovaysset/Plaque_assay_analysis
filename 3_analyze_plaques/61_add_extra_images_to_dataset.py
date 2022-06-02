"""
Step 3. Analyze each cell of the grid to search for lysis plaque.
For each plate, the grid was previously located. Patches are extracted and a CNN is trained to make the predictions.
6.1 generate train, val, test samples + labels + grids
6.2 train (with data augmentation) and evaluate the model
"""

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
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import find_contours
from skimage.restoration import rolling_ball
from scipy.ndimage import binary_fill_holes
from scipy.stats import mode

import napari

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_farid"

target_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/train_set2"
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

grids = np.load("D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_grids.npy")

def pad_image(im, target_shape=(96, 96)):
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
        return im_with_defined_shape.astype("uint8"), 0
    elif dx > 0:
        pad_x = np.zeros((dx, im.shape[1]))
        im_with_defined_shape = np.concatenate([im, pad_x], axis=0)
        return im_with_defined_shape[:, :target_shape[1]].astype("uint8"), 1
    elif dy > 0:
        pad_y = np.zeros((im.shape[0], dy))
        im_with_defined_shape = np.concatenate([im, pad_y], axis=1)
        return im_with_defined_shape[:target_shape[0], :].astype("uint8"), 2
    else:
        return im[:target_shape[0], :target_shape[1]].astype("uint8"), 3

images_to_add = ["spotassay_2b.png",
                "spotassay_9d.png",
                "spotassay_035_01.jpg",
                "spotassay_72c.png",
                "spotassay_116_01.jpg",
                "spotassay_118_03.jpg",
                "spotassay_269_01.jpg",
                "spotassay_291_01.jpg",
                "spotassay_137_01.jpg",
                "spotassay_264_01.jpg"]

grid_size = (96, 96)
images, names = [], []
for k, ifl in enumerate(sorted(os.listdir(directory))[:]):

    if ifl in images_to_add:

        print(f"Image {ifl}")
        im = imageio.imread(directory + "/" + ifl)
        if len(im.shape) == 2:
            width, height = im.shape
        else:
            width, height = im.shape[0], im.shape[1]
            im = im[:, :, 0]

        patch_x, patch_y = grid_size
        grid = grids[k]

        for l, coord in enumerate(grid):
            x, y = coord
            top, bottom, left, right = np.max([0, x - patch_x // 2]), np.min([x + patch_x  // 2, height]), np.max([0, y - patch_y  // 2]), np.min([width, y + patch_y  // 2])
            patch = im[top:bottom, left:right]

            if patch.shape[0] < grid_size[0] * 0.60 or patch.shape[1] < grid_size[1] * 0.60:
                continue
            elif patch.shape[0] < grid_size[0] or patch.shape[1] < grid_size[1]:
                patch, idx = pad_image(patch, target_shape=grid_size)

            if len(str(l)) == 1:
                im_idx = "0" + str(l)
            else:
                im_idx = str(l)

            extension = ifl.split(".")[-1]
            name = ifl.split(".")[0] + "_" + im_idx + "." + extension

            imageio.imwrite(target_dir + "/" + name, patch)
