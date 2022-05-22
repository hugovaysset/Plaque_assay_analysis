import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import cv2
from skimage.filters import threshold_otsu, prewitt
from skimage.morphology import binary_dilation, binary_erosion
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes

path = "D:\Documents\Thèse\Projet_Coli/image_analysis\Réplicat 1\grey/spotassay_18a.png"
im = imageio.imread(path)

im = prewitt(im)
thresh = threshold_otsu(im)
im_bin = binary_erosion(binary_fill_holes(im > thresh))

plt.imshow(im_bin)
plt.show()
