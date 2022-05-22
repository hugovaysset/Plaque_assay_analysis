"""
The aim of this script is to split each image containing n plates (usually 2 or 4)
into n images of a single plate. This step is important to make the subsequent processing
steps easier.
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes

directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/Spotassay_ines_photos"

# out directories
single_plates_rgb_dir = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/rgb"  # where to save each rgb image
if not os.path.isdir(single_plates_rgb_dir):
    os.mkdir(single_plates_rgb_dir)
  
single_plates_grey_dir = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey"  # where to save each grey scale image
if not os.path.isdir(single_plates_grey_dir):
    os.mkdir(single_plates_grey_dir)

# where to put the problematic files (for which extraction failed...)
issues_dir = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/issues"
if not os.path.isdir(issues_dir):
    os.mkdir(issues_dir)

success, failure = 0, 0

for k, ifl in enumerate(os.listdir(directory)):
    extension = ifl.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 10 == 0:
            dir_size = os.listdir(directory)
            print(f"Extracting image {k}/{len(dir_size)}.")
        im_rgb = imageio.imread(directory + "/" + ifl)
        im = im_rgb[:, :, 0]  # go to grey scale for the preprocessing

        # preprocess the image : binarize, dilate and fill holes
        threshold = threshold_otsu(im)
        im_bin = im > threshold

        for i in range(1):
            im_bin = binary_dilation(im_bin)
        im_bin = (np.ones(im_bin.shape) - im_bin) * 255

        im_bin = binary_fill_holes(im_bin)

        # get the contours of each plate
        contours_prov = find_contours(im_bin)
        contours = []
        for contour in contours_prov:
            if cv2.contourArea(np.array(contour, dtype="float32")) > 100000:
                contours.append(np.array(contour, dtype="float32"))

        #  GARDEFOU#1 : pas de plaque detectee...
        if len(contours) == 0:
            print(f"No contour for experiment {ifl}. Extraction FAILED.")
            shutil.move(directory + "/" + ifl, issues_dir + "/" + ifl)
            failure += 1
            continue

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # extract the are of the image corresponding to each plate and save it in a dedicated image
            if len(ifl.split("_")) > 2:  # images with name spotassay_Xa_Xb or spotassay_Xc_X_d
                    print("Two plates")
                    out_file_name = ifl.strip(f".{extension}").split("_")[0] + "_" + ifl.strip(f".{extension}").split("_")[idx+1] + "." + extension
            else:  # images with four plates in a random order
                print("Four plates")
                out_file_name = ifl.strip(f".{extension}").split("_")[0] + "_" + ifl.strip(f".{extension}").split("_")[1] + "_" + str(idx) + "." + extension
            
            # GARDEFOU#2 : L'image d'une plaque devrait etre carree environ, sinon elle a ete mal decoupee (plaque fusionnees...)
            if abs(1 - h/w) > 0.5:
                shutil.move(directory + "/" + ifl, issues_dir + "/" + ifl)
                print("All plates were probably merged for experiment")
                failure += 1
                continue       
            
            imageio.imwrite(single_plates_rgb_dir + "/" + out_file_name, im_rgb[x:x+w, y:y+h, :])
            imageio.imwrite(single_plates_grey_dir + "/" + out_file_name, im[x:x+w, y:y+h])
        success += 1

print(f"{success}/{success+failure} successess")

