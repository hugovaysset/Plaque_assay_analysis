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

def crop_empty_sides(image):
    """
    Crop the left/right/bottom/top parts of the image corresponding to empty (white areas)
    im (array): grey scale image of the plaque
    """
    height, width = image.shape
    top, bottom, left, right = 0, height-1, 0, width-1
    white_pixel_value, alpha = 255, 0.95

    while np.sum(image[top, :]) >= white_pixel_value * alpha * width: #3/4 of the row are white pixels = empty area
        top += 1

    while np.sum(image[bottom, :]) >= white_pixel_value * alpha * width: #3/4 of the row are white pixels = empty area
        bottom -= 1

    while np.sum(image[:, left]) >= white_pixel_value * alpha * height: #3/4 of the row are white pixels = empty area
        left += 1

    while np.sum(image[:, right]) >= white_pixel_value * alpha * height: #3/4 of the row are white pixels = empty area
        right -= 1

    return top, bottom, left, right

# out directories
single_plates_rgb_dir = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/rgb"  # where to save each rgb image
if not os.path.isdir(single_plates_rgb_dir):
    os.mkdir(single_plates_rgb_dir)
  
single_plates_grey_dir = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey"  # where to save each grey scale image
if not os.path.isdir(single_plates_grey_dir):
    os.mkdir(single_plates_grey_dir)

n_two_plates, n_four_plates = 0, 0
for k, ifl in enumerate(os.listdir(directory)):
    extension = ifl.split(".")[-1]  # get the file extension

    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 10 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")
        
        im_rgb = imageio.imread(directory + "/" + ifl)
        im = im_rgb[:, :, 0]  # go to grey scale for the preprocessing

        # detect two or four plates
        height, width = im.shape
        if len(ifl.strip(f".{extension}").split("_")) > 2: # two plates in the image
            n_two_plates += 1
            for idx in range(2):
                out_file_name = ifl.strip(f".{extension}").split("_")[0] + "_" + ifl.strip(f".{extension}").split("_")[idx+1] + "." + extension 

                top, bottom, left, right = crop_empty_sides(im)

                imageio.imwrite(single_plates_rgb_dir + "/" + out_file_name, im_rgb[np.max(top, idx*width//2):np.min(bottom, (idx+1)*width//2), left:right, :])
                imageio.imwrite(single_plates_grey_dir + "/" + out_file_name, im[np.max(top, idx*width//2):np.min(bottom, (idx+1)*width//2), left:right])
        else: # four plates in the image
            n_four_plates += 1
            
            top, bottom, left, right = crop_empty_sides(im)
            
            # im1 : top-left
            out_file_name = ifl.strip(f".{extension}") + "_0" + str(0) + "." + extension 
            imageio.imwrite(single_plates_rgb_dir + "/" + out_file_name, im_rgb[top:height//2:, left:width//2, :])
            imageio.imwrite(single_plates_grey_dir + "/" + out_file_name, im[top:height//2:, left:width//2])
            
            # im2 : top-right
            out_file_name = ifl.strip(f".{extension}") + "_0" + str(1) + "." + extension 
            imageio.imwrite(single_plates_rgb_dir + "/" + out_file_name, im_rgb[top:height//2, left:height//2, :])
            imageio.imwrite(single_plates_grey_dir + "/" + out_file_name, im[top:height//2, left:height//2])
            
            # im3 : bottom-left
            out_file_name = ifl.strip(f".{extension}") + "_0" + str(2) + "." + extension 
            imageio.imwrite(single_plates_rgb_dir + "/" + out_file_name, im_rgb[height//2:, height//2:height, :])
            imageio.imwrite(single_plates_grey_dir + "/" + out_file_name, im[:width//2, height//2:height])
            
            # im4 : bottom-right
            out_file_name = ifl.strip(f".{extension}") + "_0" + str(3) + "." + extension 
            imageio.imwrite(single_plates_rgb_dir + "/" + out_file_name, im_rgb[width//2:width, height//2:height, :])
            imageio.imwrite(single_plates_grey_dir + "/" + out_file_name, im[width//2:width, height//2:height])
        
print(f"N_two_plates : {n_two_plates}")
print(f"N_four_plates : {n_four_plates}")



directory = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1\grey/qr_code"
save_dir = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1\grey_crop"

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for k, path in enumerate(os.listdir(directory)):
    
    extension = path.split(".")[-1]  # get the file extension
    if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
        if k % 100 == 0:
            dir_size = len(os.listdir(directory))
            print(f"Extracting image {k}/{dir_size}.")
        
        im = imageio.imread(directory + "/" + path)

        top, bottom, left, right = crop_empty_sides(im)

        imageio.imwrite(save_dir + "/" + path, im[top:bottom, left:right])
