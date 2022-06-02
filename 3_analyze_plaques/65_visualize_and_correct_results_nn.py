import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

import napari

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_farid"
predictions_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/predictions_bin"
grids_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grids"

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

names, stack, predictions, polygons, grids = [], [], [], [], []
grid_size = (96, 96)
final_im_shape = (1500, 1500)
k = 0
init_idx, n_images = 250, 100
for f in sorted(os.listdir(predictions_dir))[init_idx:init_idx+n_images]:

    if f.endswith(".csv"):
        if k % 100 == 0:
            dir_size = os.listdir(directory)
            print(f"Extracting image {k}/{len(dir_size)}.")
    
        ifl = f[:-4]
        names.append(f)

        # pad image
        if os.path.isfile(directory + "/" + ifl + ".jpg"):
            im = imageio.imread(directory + "/" + ifl + ".jpg")
        elif os.path.isfile(directory + "/" + ifl + ".png"):
            im = imageio.imread(directory + "/" + ifl + ".png")
        if len(im.shape) > 2:
            im = im[:, :, 0]
        if im.shape[0] < final_im_shape[0] or im.shape[1] < final_im_shape[1]:
            im, xx = pad_image(im, final_im_shape)
        stack.append(im)


        # get predictions
        preds = pd.read_csv(predictions_dir + "/" + f, sep=";")
        predictions.append(preds["predict"].to_numpy())

        # get grid
        grid = pd.read_csv(grids_dir + "/" + ifl + ".csv", sep=";")
        grid_xy = grid[["X", "Y"]].to_numpy()

        grids.append(grid_xy)

        polygon = []
        for l, (g, p) in enumerate(zip(grid_xy, preds["predict"].to_numpy())):
            if p > 0:  # plage de lyse detectee
                i, j = g
                polygon.append(np.array([[k, i - 50, j - 50],
                                         [k, i - 50, j + 50],
                                         [k, i + 50, j + 50],
                                         [k, i + 50, j - 50],]))
            else:
                polygon.append(np.array([[k, 0, 0], [k, 0, 0], [k, 0, 0], [k, 0, 0]]))
        polygons.append([p for p in polygon])

        k += 1

names = np.array(names)
stack = np.array(stack)
predictions = np.array(predictions)
polygons = np.array(polygons).reshape(names.shape[0] * 96, 4, 3)
grids = np.array(grids)

assert names.shape[0] == stack.shape[0] \
        and stack.shape[0] == predictions.shape[0]

print(names.shape, stack.shape, predictions.shape, polygons.shape)

COLORS = ["red", "#00aa00ff", "blue"]

viewer = napari.Viewer()
im_layer = viewer.add_image(stack)
# p_layer = viewer.add_shapes(polygons, shape_type='polygon', edge_color=predictions, edge_color_cycle=COLORS, edge_width=5, face_color="#ffffff00")
p_layer = viewer.add_shapes(polygons, shape_type='polygon', properties={"label": predictions.reshape(-1)}, 
                            edge_color="label", edge_color_cycle=COLORS, edge_width=5, face_color="#ffffff00")

def get_closer_item_index(x, y, grid):
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    distances = [euclidean_distance(x, y, x1, x2) for (x1, x2) in grid]
    return np.argmin(distances)

def retrieve_grids_at_specific_frame(grids, t):
    ret = []
    for g in grids:
        if g[0, -1] == t:
            ret.append(g)
    return ret

global did_correct
did_correct = np.zeros(stack.shape[0])

@p_layer.bind_key("0")
def prediction_to_zero(layer):
    z, x, y = layer.coordinates
    z, x, y = int(z), int(x), int(y)

    closer_idx = get_closer_item_index(x, y, grids[z])
    
    print(f"Updating plage {(closer_idx // 12 + 1, closer_idx % 8 + 1)}")
    predictions[z, closer_idx] = 0
    print(f"New prediction : {0}")
    print("\n")

    did_correct[z] = 1

    print(predictions[z])

    polygons[z * 96 + closer_idx] = np.array([[z, 0, 0], [z, 0, 0], [z, 0, 0], [z, 0, 0]])
    layer.add(polygons[z * 96 + closer_idx])

@p_layer.bind_key("1")
def prediction_to_one(layer):
    z, x, y = layer.coordinates
    z, x, y = int(z), int(x), int(y)

    closer_idx = get_closer_item_index(x, y, grids[z])
    
    print(f"Updating plage {(closer_idx // 12 + 1, closer_idx % 8 + 1)}")
    predictions[z, closer_idx] = 1
    print(f"New prediction : {1}")
    print("\n")

    print(predictions[z])

    i, j = grids[z, closer_idx]
    polygons[z * 96 + closer_idx] = np.array([[z, i - 50, j - 50],
                                              [z, i - 50, j + 50],
                                              [z, i + 50, j + 50],
                                              [z, i + 50, j - 50],])

    did_correct[z] = 1

    layer.add(polygons[z * 96 + closer_idx])

@p_layer.bind_key("2")
def prediction_to_two(layer):
    z, x, y = layer.coordinates
    z, x, y = int(z), int(x), int(y)

    closer_idx = get_closer_item_index(x, y, grids[z])
    
    print(f"Updating plage {(closer_idx // 12 + 1, closer_idx % 8 + 1)}")
    predictions[z, closer_idx] = 2
    print(f"New prediction : {2}")
    print("\n")

    i, j = grids[z, closer_idx]
    polygons[z * 96 + closer_idx] = np.array([[z, i - 50, j - 50],
                                              [z, i - 50, j + 50],
                                              [z, i + 50, j + 50],
                                              [z, i + 50, j - 50],])

    did_correct[z] = 1

    layer.add(polygons[z * 96 + closer_idx])

@p_layer.bind_key("n")
def set_all_zeros(layer):
    z, x, y = layer.coordinates
    z, x, y = int(z), int(x), int(y)

    predictions[z] = 0

    did_correct[z] = 1

    print(predictions[z])

    for closer_idx in range(96):
        polygons[z * 96 + closer_idx] = np.array([[z, 0, 0], [z, 0, 0], [z, 0, 0], [z, 0, 0]])
        layer.add(polygons[z * 96 + closer_idx])

@p_layer.bind_key("s")
def save_predictions(layer):
    z, x, y = layer.coordinates
    z = int(z)

    print(f"Saving image... {names[z]}")
    df = pd.DataFrame(predictions[z], dtype="uint8", columns=["predict"])
    df["file"] = np.array([names[z] for i in range(predictions.shape[1])])
    df.to_csv(predictions_dir + "/" + names[z], sep=";")

    did_correct[z] = 0  # reset the did_correct attribute since has been saved

@p_layer.bind_key("m")
def save_all(layer):
    """
    Save all the predictions which have been modified at once
    """
    z, x, y = layer.coordinates
    z = int(z)

    did_correct = np.ones(stack.shape[0])  #TODO: fix this shit

    to_save = np.nonzero(did_correct)

    for name, pred in zip(names[to_save], predictions[to_save]):
        df = pd.DataFrame(pred, dtype="uint8", columns=["predict"])
        df["file"] = np.array([name for i in range(predictions.shape[1])])
        df.to_csv(predictions_dir + "/" + name, sep=";")
    
    did_correct = np.zeros(stack.shape[0])

    print(f"Saved {to_save[0].shape[0]} predictions at once.")

napari.run()
