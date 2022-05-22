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

save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_starting_point1.csv"

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

if table_path == "":
    names, sp, issues = [], [], []
else:
    save = pd.read_csv(table_path, sep=";")
stack = []
for k, ifl in enumerate(os.listdir(directory)[:]):
    extension = ifl.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 100 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")

        im = imageio.imread(directory + "/" + ifl)

        if im.ndim > 2:
            im = im[:, :, 0]

        if table_path == "":
            names.append(ifl)
            sp.append(np.array([k, 0, 0]))
            issues.append(False)

        stack.append(pad_image(im))

if table_path == "":
    sp = np.array(sp)
    names = np.array(names)
    issues = np.array(issues)
    starting_points = pd.DataFrame(columns=["image", "X", "Y", "issues"])
else:
    names = save["image"].to_numpy()
    sp = np.vstack([[k for k in range(save.shape[0])], save["X"].to_numpy(), save["Y"].to_numpy()]).T
    issues = save["issues"].to_numpy()

stack = np.array(stack)

viewer = napari.view_image(stack)
points_layer = viewer.add_points(sp, face_color="red", edge_color="black", size=10)

@points_layer.mouse_drag_callbacks.append
def click_drag(layer, event):
    dragged = False
    yield
    # on move
    while event.type == 'mouse_move':
        dragged = True
        yield
    # on release
    if not dragged:
        k, x, y = layer.coordinates
        k, x, y = int(k), int(x), int(y)

        assert (names[k] == save.iloc[k]["image"]), save.iloc[k]["image"]

        if 0 <= x < 500 and 0 <= y < 500:
            print(f"adding point at position ({x},{y})")
            sp[k] = [k, x, y]
            layer.data = sp

@points_layer.bind_key("s")
def save_layer(layer):
    if table_path == "":
        save = pd.DataFrame(columns=["image", "X", "Y"])
    else:
        save = pd.read_csv(table_path, sep=";")
        
    save["image"] = names
    save["X"] = sp[:, 1]
    save["Y"] = sp[:, 2]
    save["issues"] = issues
    save.to_csv(save_path, sep=";", index=False)

    print("saved file!")

@viewer.bind_key("i")
def save_layer(layer):
    k, x, y = layer.coordinates
    k, x, y = int(k), int(x), int(y)
    
    assert (names[k] == save.iloc[k]["image"])

    print(f"added issue for image {names[k]}")
    
    issues[k] = True

napari.run()
