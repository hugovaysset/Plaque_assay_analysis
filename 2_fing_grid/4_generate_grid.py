"""
Step 2. Set the grid for each image.
After the preprocessing step, now we want to identify the coordinates of each phage-bacteria combination
in the plates, to assign a measured result (kill or not kill...) to a specific phage-bact couple.
The first step to find the grid is to get the starting point (upper-left) in the grid. 
"""


from traceback import StackSummary
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

import napari

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_farid"

table_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_starting_point.csv"

save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_grids.npy"

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

starting_points = pd.read_csv(table_path, sep=";")

grid_size = (12, 8)
grid_stride = (105, 105)
stack, grids = [], []
for k, ifl in enumerate(os.listdir(directory)[:]):
    extension = ifl.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 100 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")

        im = imageio.imread(directory + "/" + ifl)

        x0, y0 = starting_points[starting_points["image"] == ifl][["X", "Y"]].to_numpy()[0]

        if im.ndim > 2:
            im = im[:, :, 0]

        g = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                g.append(np.array([x0 + i * grid_stride[0], y0 + j * grid_stride[1]]))

        grids.append(np.array(g))

grids = np.array(grids)

np.save(save_path, grids)
