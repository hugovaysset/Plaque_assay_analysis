import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

import cv2
from skimage.filters import farid_h, farid_v
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes

directory = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1\grey_crop"

save_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1\grey_farid"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

n_two_plates, n_four_plates = 0, 0
failure = 0
for k, ifl in enumerate(sorted(os.listdir(directory))):
    extension = ifl.split(".")[-1]  # get the file extension
    
    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"] and ifl == "spotassay_075_03.jpg":
        if k % 10 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")
        
        im = imageio.imread(directory + "/" + ifl)

        height, width = im.shape
        top, bottom, left, right = 0, height, 0, width

        im_far_h = farid_h(im)  # detect horizontal edges
        im_far_v = farid_v(im)  # detect vertical edges
        cum_far_v = np.sum(np.abs(farid_v(im)), axis=0)
        cum_far_h = np.sum(np.abs(farid_h(im)), axis=1)

        thresh, i, under_threshold = 0.005 * width, 0, 0
        while under_threshold < 10 and i < width:  # 10 colonnes d'affilee avec une somme < thresh
            if cum_far_v[i] < thresh:
                under_threshold += 1
            else:
                under_threshold = 0
            i += 1
        left = i

        thresh, j, under_threshold_h = 0.005 * height, 0, 0
        while under_threshold_h < 10 and j < height:  # 10 colonnes d'affilee avec une somme < thresh
            if cum_far_h[j] < thresh:
                under_threshold_h += 1
            else:
                under_threshold_h = 0
            j += 1
        top = j

        thresh, k, under_threshold_hrev = 0.005 * height, height-1, 0
        while under_threshold_hrev < 10 and k >= 0:  # 10 colonnes d'affilee avec une somme < thresh
            if cum_far_h[k] < thresh:
                under_threshold_hrev += 1
            else:
                under_threshold_hrev = 0
            k -= 1
        bottom = k

        thresh, l, under_threshold_vrev = 0.005 * width, width-1, 0
        while under_threshold_vrev < 10 and l >= 0:  # 10 colonnes d'affilee avec une somme < thresh
            if cum_far_v[l] < thresh:
                under_threshold_vrev += 1
            else:
                under_threshold_vrev = 0
            l -= 1
        right = l

        print(top, bottom, left, right)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im)
        ax[1].imshow(im[top:bottom, left:right])
        plt.show()

        break

        # if (bottom - top) < height * 0.8 and (right - left) >= width * 0.8:
        #     imageio.imwrite(save_dir + "/" + ifl, im[:, left:right])
        #     failure += 1
        # elif (bottom - top) >= height * 0.8 and (right - left) < width * 0.8:
        #     imageio.imwrite(save_dir + "/" + ifl, im[top:bottom, :])
        #     failure += 1
        # elif (bottom - top) >= height * 0.8 and (right - left) >= width * 0.8:
        #     imageio.imwrite(save_dir + "/" + ifl, im[top:bottom, left:right])
        # else:
        #     imageio.imwrite(save_dir + "/" + ifl, im)
        #     failure += 1

print(f"#fails : {failure}") 