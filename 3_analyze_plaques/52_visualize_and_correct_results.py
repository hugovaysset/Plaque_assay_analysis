import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

import napari

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_farid"

save_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/predictions"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

names, stack, origins, grids, predictions, polygons = [], [], [], [], [], []
stride_x, stride_y = 105, 105
final_im_shape = (1500, 1500)
for k, f in enumerate(sorted(os.listdir(directory))[:25]):
    
    extension = f.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 10 == 0:
            dir_size = os.listdir(directory)
            print(f"Extracting image {k}/{len(dir_size)}.")
    
        ifl = f.strip(f".{extension}")
        names.append(ifl)

        im = imageio.imread(directory + "/" + f)

        dx, dy = final_im_shape[0] - im.shape[0], final_im_shape[1] - im.shape[1]
        pad_x = np.zeros((dx, im.shape[1]))
        pad_y = np.zeros((final_im_shape[0], dy))
        im_with_defined_shape = np.concatenate([im, pad_x], axis=0)
        im_with_defined_shape = np.concatenate([im_with_defined_shape, pad_y], axis=1)
        stack.append(im_with_defined_shape)

        origin = np.array([56, 134])
        origins.append(origin)
        grid = np.array([[origin[0] + i * stride_x, origin[1] + j * stride_y] for i in range(0, 12) for j in range(0, 8)])
        grids.append(grid) 

        if k % 2 == 0:
            prediction = np.zeros(12*8)
        else:
            prediction = np.ones(12*8)
        predictions.append(prediction)

        polygon = []
        for l, (g, p) in enumerate(zip(grid, prediction)):
            if p > 0:  # plage de lyse detectee
                i, j = g
                polygon.append(np.array([[k, i - 50, j - 50],
                                        [k, i - 50, j + 50],
                                        [k, i + 50, j + 50],
                                        [k, i + 50, j - 50],]))
            else:
                polygon.append(np.array([[k, 0, 0], [k, 0, 0], [k, 0, 0], [k, 0, 0]]))
        polygons.append([p for p in polygon])

names = np.array(names)
stack = np.array(stack)
origins = np.array(origins)
grids = np.array(grids)
predictions = np.array(predictions)
polygons = np.array(polygons).reshape(10 * 96, 4, 3)

viewer = napari.Viewer()
im_layer = viewer.add_image(stack)
p_layer = viewer.add_shapes(polygons, shape_type='polygon', edge_color="red", edge_width=5, face_color="#ffffff00")

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

@p_layer.bind_key("0")
def prediction_to_zero(layer):
    z, x, y = layer.coordinates
    z, x, y = int(z), int(x), int(y)

    closer_idx = get_closer_item_index(x, y, grids[z])
    
    print(f"Updating plage {(closer_idx // 12 + 1, closer_idx % 8 + 1)}")
    predictions[z, closer_idx] = 0
    print(f"New prediction : {0}")
    print("\n")

    print(predictions[z])

    polygons[z * 96 + closer_idx] = np.array([[z, 0, 0], [z, 0, 0], [z, 0, 0], [z, 0, 0]])
    layer.data = polygons

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
    layer.data = polygons

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
    layer.data = polygons

@p_layer.bind_key("s")
def save_predictions(layer):
    z, x, y = layer.coordinates
    z = int(z)

    print(f"Saving image... {names[z]}")
    df = pd.DataFrame(predictions[z], dtype="uint8")
    df.to_csv(save_dir + "/" + names[z] + ".csv", sep=";")

napari.run()
