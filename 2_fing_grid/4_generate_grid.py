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

table_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_starting_point1.csv"

# save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_grids.npy"
save_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grids"

starting_points = pd.read_csv(table_path, sep=";")

grid_size = (12, 8)
grid_stride = (105, 105)
stack, grids = [], []
for k, ifl in enumerate(sorted(os.listdir(directory))):
    extension = ifl.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 100 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")

        im = imageio.imread(directory + "/" + ifl)

        x0, y0 = starting_points[starting_points["image"] == ifl][["X", "Y"]].to_numpy()[0]

        f_name = ifl[:-4]

        if im.ndim > 2:
            im = im[:, :, 0]

        g = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                g.append({"file": ifl, "X": x0 + i * grid_stride[0], "Y": y0 + j * grid_stride[1]})

        df = pd.DataFrame(g)
        df.to_csv(save_dir + "/" + f_name + ".csv", sep=";")
