import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

import napari

directory = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/train_set2"
save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/train_set_additional.csv"

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

images, names, annotations, points = [], [], [], []
for k, f in enumerate(sorted(os.listdir(directory))[:]):

    extension = f.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 500 == 0:
            dir_size = os.listdir(directory)
            print(f"Extracting image {k}/{len(dir_size)}.")

        ifl = f.strip(f".{extension}")
        im = imageio.imread(directory + "/" + f)

        names.append(f)
        images.append(pad_image(im))
        annotations.append(0)
        points.append(np.array([k, 0, 0]))

names = np.array(names)
images = np.array(images)
annotations = np.array(annotations)
points = np.array(points)

print(images.shape)

COLORS = ["red", "#00aa00ff", "blue"]

viewer = napari.Viewer()
im_layer = viewer.add_image(images)
# p_layer = viewer.add_points(points, properties={"label": annotations}, edge_color="label", edge_color_cycle=COLORS, face_color="label", face_color_cycle=COLORS)
    
@viewer.mouse_drag_callbacks.append
def prediction_to_zero(viewer, event):
    z, x, y = np.round(viewer.layers[0].coordinates).astype("uint16")
    if "Alt" in event.modifiers:
        print(z)
        print(f"Updating images {names[z]} to label 0.")
        annotations[z] = 0
    elif "Shift" in event.modifiers:
        print(f"Updating images {names[z]} to label 2.")
        annotations[z] = 2
    else:
        print(f"Updating images {names[z]} to label 1.")
        annotations[z] = 1

# @im_layer.bind_key("1")
# def prediction_to_zero(layer):
#     z = viewer.dims.events.current_step

#     print(f"Updating images {names[z]} to label 1.")

#     annotations[z] = 1

# @viewer.bind_key("2")
# def prediction_to_zero(layer):
#     z = viewer.dims.events.current_step

#     print(f"Updating images {names[z]} to label 2.")

#     annotations[z] = 2

@viewer.bind_key("s")
def save_predictions(layer):

    print(f"Saving labels...")
    df = pd.DataFrame(columns=["file", "gt"])
    df["file"] = names
    df["gt"] = annotations

    df.to_csv(save_path, sep=";")
    print("Saved!\n")

napari.run()
