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

directory = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/train_set"
labels_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/train_set.csv"

def get_lenet(input_shape, plot_summary=False):
    lenet = keras.Sequential()

    lenet.add(layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation='relu', input_shape=input_shape))
    lenet.add(layers.MaxPooling2D())

    lenet.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'))
    lenet.add(layers.MaxPooling2D())
    
    lenet.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
    lenet.add(layers.MaxPooling2D())

    lenet.add(layers.Flatten())

    lenet.add(layers.Dense(units=32, activation='relu'))

    lenet.add(layers.Dense(units=16, activation='relu'))
    
    lenet.add(layers.Dense(units=8, activation='relu'))

    lenet.add(layers.Dense(units=4, activation='relu'))

    lenet.add(layers.Dense(units=3, activation = 'softmax'))  # 3 classes configuration
    # lenet.add(layers.Dense(units=1, activation = 'sigmoid'))  # binary configuration
    
    if plot_summary:
        print(lenet.summary())

    # lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lenet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lenet

def one_hot_encode(ar):
    l = []
    for k, elt in enumerate(ar):
        if elt == 0:
            l.append(np.array([1, 0, 0]))
        elif elt == 1:
            l.append(np.array([0, 1, 0]))
        elif elt == 2:
            l.append(np.array([0, 0, 1]))
        else:
            l.append([np.nan, np.nan, np.nan])
    return np.array(l)

def binarize(label):
    return int(label != 0)

binarize = np.vectorize(binarize)

labels = pd.read_csv(labels_path, sep=";")

images, gt = [], []
for k, f in enumerate(sorted(os.listdir(directory))[:]):

    extension = f.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 500 == 0:
            dir_size = os.listdir(directory)
            print(f"Extracting image {k}/{len(dir_size)}.")

        ifl = f.strip(f".{extension}")
        im = imageio.imread(directory + "/" + f)
        images.append((im - im.min()) / (im.max() - im.min()))  # Normalize images
        gt.append(labels[labels["file"] == f]["gt"])

# get data
images = np.expand_dims(np.array(images), axis=-1)
gt = one_hot_encode(np.array(gt).squeeze(-1))
# gt = binarize(np.array(gt).squeeze(-1))

print(images.shape, gt.shape)
   
# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
        rotation_range = 60,
        # shear_range = 0.1,
        # zoom_range = 0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # brightness_range = (0.9, 1.1),
        # fill_mode="nearest"
        )

# i = 0
# for batch, gts in datagen.flow(images, gt, shuffle=True, batch_size=4):
#     i += 1
#     fig, ax = plt.subplots(1, 4)
#     for j in range(4):
#         ax[j].imshow(batch[j].squeeze(-1))
#         ax[j].set_title(gts[j])
#     plt.show()

#     if i > 10:
#         break

# assert False

# # visualize predictions
# idx_min, idx_max = 151, 170
# fig, axes = plt.subplots(1, idx_max - idx_min, figsize=(20, 5))
# for ax, im, lab in zip(axes, images[idx_min:idx_max], gt[idx_min:idx_max]):
#     ax.imshow(im.squeeze(-1))
#     ax.set_title(lab)
# plt.show()

# get model
classifier = get_lenet((96, 96, 1), plot_summary=True)

print(classifier.summary())

fig, ax = plt.subplots(1, 2, figsize=(20, 8))

init_lr, n_epochs, batch_size = 0.01, 130, 32

history = classifier.fit(datagen.flow(images, gt, shuffle=True, batch_size=32), 
                         epochs=n_epochs, 
                         steps_per_epoch=12, #images.shape[0] // batch_size,
                         verbose=1)  #, class_weight={0: 1.1, 1: 20, 2: 44})

# plot learning curves
ax[0].plot(history.history["loss"][:], "orange", label="loss")
ax[0].legend()
ax[0].set_title("Training curve")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel(f"Loss (Categorical CE)")

ax[1].plot(history.history["accuracy"][:], "orange", label="accuracy")
ax[1].legend()
ax[1].set_title("Training curve")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

# save model
n_train_images = images.shape[0]
model_name = f"lenet_n_epochs={n_epochs}_init_lr={init_lr}_3convblocks_4dense_n_train_images={n_train_images}"
model_save_dir = f"D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/models"

# if os.path.isdir(model_save_dir + "/" + model_name):
#     model_name = model_name + "_" + str(datetime.datetime.now())

classifier.save(model_save_dir + "/" + model_name)

plt.show()
plt.savefig(model_save_dir + "/" + model_name + "/" + "learning_curves.png")

#### Test data
test_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/test_set"
predictions_save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/test_pred.csv"
predictions_bin_save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/grey_nn/test_pred_bin.csv"


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

# get data
test_set = np.expand_dims(np.array(test_set), axis=-1)
test_names = np.array(test_names)

predictions = classifier.predict(test_set)

# save raw predictions
preds_table = pd.DataFrame(columns=["file", "predict_0", "predict_1", "predict_2"])
preds_table["file"] = test_names
preds_table["predict_0"] = predictions[:, 0]
preds_table["predict_1"] = predictions[:, 1]
preds_table["predict_2"] = predictions[:, 2]
preds_table.to_csv(predictions_save_path, sep=";")

# save predicted class
preds_table_bin = pd.DataFrame(columns=["file", "predict"])
preds_table_bin["file"] = test_names
preds_table_bin["predict"] = predictions
preds_table_bin.to_csv(predictions_bin_save_path, sep=";")

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
