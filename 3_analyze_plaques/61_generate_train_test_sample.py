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

train_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/train_set"
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

val_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/val_set"
if not os.path.isdir(val_dir):
    os.mkdir(val_dir)

test_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/test_set"
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

grids = np.load("D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/3_grids.npy")
grid_size = (96, 96)
# generate the datasets

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

if len(os.listdir(train_dir)) == 0:
    indices = np.arange(len(os.listdir(directory)))
    np.random.shuffle(indices)

    train_idx, val_idx, test_idx = indices[:20], indices[20:30], indices[30:40]
    train_set, val_set, test_set = np.array(sorted(os.listdir(directory)))[train_idx],  np.array(sorted(os.listdir(directory)))[val_idx], np.array(sorted(os.listdir(directory)))[test_idx]
    train_grids, val_grids, test_grids = grids[train_idx], grids[val_idx], grids[test_idx]

    for dir, compo, grids in zip([train_dir, val_dir, test_dir], [train_set, val_set, test_set], [train_grids, val_grids, test_grids]):

        for k, (ifl, grid) in enumerate(zip(compo, grids)):

            print(f"Image {k}")
            im = imageio.imread(directory + "/" + ifl)
            if len(im.shape) == 2:
                width, height = im.shape
            else:
                width, height = im.shape[0], im.shape[1]
                im = im[:, :, 0]

            patch_x, patch_y = grid_size

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

                imageio.imwrite(dir + "/" + name, patch)
