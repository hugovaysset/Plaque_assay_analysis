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
test_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/val_set"
model_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/models/lenet_n_epochs=130_init_lr=0.01_3convblocks_4dense_n_train_images=371"

predictions_save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/val_pred.csv"
predictions_bin_save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/val_pred_bin.csv"

# get the test data
test_set, test_names = [], []
for k, f in enumerate(sorted(os.listdir(test_dir))[:]):

    extension = f.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 500 == 0:
            dir_size = os.listdir(test_dir)
            print(f"Extracting image {k}/{len(dir_size)}.")

        ifl = f.strip(f".{extension}")
        im = imageio.imread(test_dir + "/" + f)
        test_set.append((im - im.min()) / (im.max() - im.min()))  # Normalize image
        test_names.append(f)

test_set = np.expand_dims(np.array(test_set), axis=-1)
test_names = np.array(test_names)

# get the model
classifier = keras.models.load_model(model_path)

# perform inference
predictions = classifier.predict(test_set)

preds_table = pd.DataFrame(columns=["file", "predict_0", "predict_1", "predict_2"])
preds_table["file"] = test_names
preds_table["predict_0"] = predictions[:, 0]
preds_table["predict_1"] = predictions[:, 1]
preds_table["predict_2"] = predictions[:, 2]

preds_table.to_csv(predictions_save_path, sep=";")

# save predicted class
preds_table_bin = pd.DataFrame(columns=["file", "predict"])
preds_table_bin["file"] = test_names
preds_table_bin["predict"] = np.argmax(predictions, axis=1)

preds_table_bin.to_csv(predictions_bin_save_path, sep=";")

# visualize predictions
points = np.array([np.array([k, 0, 0]) for k in range(predictions.shape[0])])

def threshold(pred, thresh=0.9):
    """
    pred in one hot format : [0, 1, 2]
    """
    amax = np.argmax(pred)
    if amax == 1 and pred[1] >= thresh: return 1
    elif amax == 2 and pred[2] >= thresh: return 2
    else: return 0

pred_thresh = np.array([threshold(line, thresh=0.9) for line in predictions])

print(predictions)

print(pred_thresh)

COLORS = ["red", "#00aa00ff", "blue"]
viewer = napari.view_image(test_set.squeeze(-1))
viewer.add_points(points, properties={"label": pred_thresh}, edge_color="label", edge_color_cycle=COLORS, face_color="label", face_color_cycle=COLORS)
napari.run()
