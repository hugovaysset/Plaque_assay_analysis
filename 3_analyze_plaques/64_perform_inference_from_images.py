import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import datetime

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import Iterator, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import Iterator, ImageDataGenerator
import tensorflow.keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import napari

# save raw predictions
test_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_farid"
model_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/models/lenet_n_epochs=130_init_lr=0.01_3convblocks_4dense_n_train_images=371"

predictions_save_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/predictions"
predictions_bin_save_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/predictions_bin"


grids = np.load("D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_grids.npy")

# get the model
classifier = keras.models.load_model(model_path)

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


def threshold(pred, thresh=0.9):
    """
    pred in one hot format : [0, 1, 2]
    """
    amax = np.argmax(pred)
    if amax == 1 and pred[1] >= thresh: return 1
    elif amax == 2 and pred[2] >= thresh: return 2
    else: return 0


# get the test data
grid_size = (96, 96)
for k, (f, grid) in enumerate(zip(sorted(os.listdir(test_dir))[500:], grids)):

    extension = f.split(".")[-1]  # get the file extension

    # if np.random.random() <= 0.05:
    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 100 == 0:
            dir_size = os.listdir(test_dir)
            print(f"Extracting image {k}/{len(dir_size)}.")

        ifl = f[:-4]
        im = imageio.imread(test_dir + "/" + f)

        if len(im.shape) == 2:
            height, width = im.shape
        else:
            height, width = im.shape[0], im.shape[1]
            im = im[:, :, 0]

        # collect all patches within the plate 
        patch_x, patch_y = grid_size
        stack = []
        for l, coord in enumerate(grid):
            x, y = coord
            top, bottom, left, right = np.max([0, x - patch_x // 2]), np.min([x + patch_x  // 2, height]), np.max([0, y - patch_y  // 2]), np.min([width, y + patch_y  // 2])
            patch = im[top:bottom, left:right]
            if patch.shape[0] < grid_size[0] or patch.shape[1] < grid_size[1]:
                patch, idx = pad_image(patch, target_shape=grid_size)
            patch = np.expand_dims((patch - patch.min()) / (patch.max() - patch.min() + 1e-6) , axis=-1)
            stack.append(patch)
        stack = np.array(stack)

        predictions = classifier.predict(stack)

        preds_table = pd.DataFrame(columns=["file", "predict_0", "predict_1", "predict_2"])
        preds_table["file"] = np.array([ifl + ".csv" for i in range(predictions.shape[0])])
        preds_table["predict_0"] = predictions[:, 0]
        preds_table["predict_1"] = predictions[:, 1]
        preds_table["predict_2"] = predictions[:, 2]

        preds_table.to_csv(predictions_save_dir + "/" + ifl + ".csv", sep=";")

        # save predicted class
        preds_table_bin = pd.DataFrame(columns=["file", "predict"])
        preds_table_bin["file"] = np.array([ifl + ".csv" for i in range(predictions.shape[0])])
        preds_table_bin["predict"] = [threshold(line, thresh=0.9) for line in predictions]

        preds_table_bin.to_csv(predictions_bin_save_dir + "/" + ifl + ".csv", sep=";")

# # visualize predictions
# points = np.array([np.array([k, 0, 0]) for k in range(predictions.shape[0])])

# def threshold(pred, thresh=0.9):
#     """
#     pred in one hot format : [0, 1, 2]
#     """
#     amax = np.argmax(pred)
#     if amax == 1 and pred[1] >= thresh: return 1
#     elif amax == 2 and pred[2] >= thresh: return 2
#     else: return 0

# pred_thresh = np.array([threshold(line, thresh=0.9) for line in predictions])

# print(predictions)

# print(pred_thresh)

# COLORS = ["red", "#00aa00ff", "blue"]
# viewer = napari.view_image(test_set.squeeze(-1))
# viewer.add_points(points, properties={"label": pred_thresh}, edge_color="label", edge_color_cycle=COLORS, face_color="label", face_color_cycle=COLORS)
# napari.run()
