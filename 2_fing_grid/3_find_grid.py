from re import search
from tokenize import single_quoted
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import napari
import os

from skimage.restoration import rolling_ball
from skimage.filters import difference_of_gaussians
from scipy.stats import mode

# # # # # # # # # # # # # # # # # # 
# Auto rotation
# # # # # # # # # # # # # # # # # # 

def find_angle(image):
    """
    Aims to detect the orientation of the crops rows
    """
    print("looking for the angle")
    angle_min = 0
    score_min = 1

    nrow, ncol = image.shape
            
    auto_angle_score_plot = []
    angles = np.arange(0, 20, 1)
    real_BW_Otsu = np.where(image > 200, 255, 0)
    coord_map = np.fliplr(np.transpose(np.nonzero(real_BW_Otsu)))
    
    for _a in angles:
        theta = np.radians(_a)
        XY_rot = np.dot(coord_map, rotation_matrix(theta))
        
        X_rot_ceil = np.ceil(XY_rot[:, 0])  # arrondi superieur
        X_rot_ceil_unique = np.unique(X_rot_ceil)
        
        score = np.shape(X_rot_ceil_unique)[0] / abs(np.max(X_rot_ceil)-np.min(X_rot_ceil)+1)
        auto_angle_score_plot += [score]
        
        if (score < score_min):
            score_min = score
            angle_min = _a
    
    angle_min_rotation_matrix = rotation_matrix(np.deg2rad(angle_min))
    
    coord_centroid_map_Rot = np.dot(coord_map,
                                    angle_min_rotation_matrix)

    return coord_map, coord_centroid_map_Rot, angle_min
    
def rotation_matrix(_theta):
    """
    Counter clock wise rotation matrix
    """
    return np.array([[np.cos(_theta), -np.sin(_theta)],
                        [np.sin(_theta),  np.cos(_theta)]])

def plot_auto_angle_rotation(coord_map, coord_centroid_map_Rot):
    """
    Plots the cloud points of the image before and after rotation
    
    You must have computed angle_min with the auto_angle2() method.
    """
    
    plt.figure()
    
    plt.scatter(coord_map[:,0],
                coord_map[:,1], s = 0.05, marker="x", label="original")
    
    plt.scatter(coord_centroid_map_Rot[:,0],
                coord_centroid_map_Rot[:,1], s = 0.05, marker="x", label="rotated")

    plt.legend()



# # # # # # # # # # # # # # # # # # # # # 
# Fourier
# # # # # # # # # # # # # # # # # # # # # 

def Compute_Power_and_Freq(_signal):    
    fourier = np.fft.fft(_signal)
    power = np.absolute(fourier/_signal.size)**2
    freq = np.fft.fftfreq(_signal.size, d=1)
    
    return power, freq

def Get_Signal_Freq(_power, _all_freq):
    nb_points = _all_freq.size
    i=0
    _max=0
    _freq_index=0
    while _all_freq[i] >= 0 and i < nb_points-1:
        
        if (_power[i+1] > _power[i]):
            if (_power[i+1] > _max):
                _max = _power[i+1]
                _freq_index = i+1
        i+=1
    if (_freq_index == 0):
        _freq_index += 1
        
    elif(_all_freq[_freq_index] < 0):
        _freq_index=1
    
    return _all_freq[_freq_index]

def Clamp_Value(_value, _min, _max):
    if _value < _min:
        _value = _min
    elif _value > _max:
        _value = _max
    return int(_value)

def Get_Corrected_Peak_Index(_histogram, _peak_index, _search_window_half_width):
    subset_low_boundary = Clamp_Value(_peak_index-_search_window_half_width,
                                      0,
                                      _histogram.size-1)
    subset_high_boundary = Clamp_Value(_peak_index+_search_window_half_width+1,
                                       0,
                                       _histogram.size-1)
    subset = _histogram[subset_low_boundary:subset_high_boundary]
    return int(np.argsort(subset)[-1] - _search_window_half_width)

def Search_Periodic_Peaks(_histogram, _period, n_max_peaks=100):
    
    signal_max_index = np.argsort(_histogram)[-1]
    n_peaks = 0

    search_window_half_width = int(0.1*_period)
    if (search_window_half_width==0):
        search_window_half_width+=1
    
    first_part_rows=[]
    peak_index = signal_max_index
    while (peak_index > 0 and n_peaks <= n_max_peaks):
        # print(peak_index)
        correction_to_global_max_index = Get_Corrected_Peak_Index(_histogram,
                                                                  peak_index,
                                                                  search_window_half_width)
        corrected_peak_index = int(peak_index + correction_to_global_max_index)
        
        if (_histogram[corrected_peak_index] > 0):
            # print(f"peak : {len(first_part_rows)} at position X={corrected_peak_index}")
            first_part_rows += [corrected_peak_index]
        
        peak_index = corrected_peak_index - _period
        n_peaks += 1
    
    second_part_rows=[]
    peak_index = signal_max_index
    while (peak_index < _histogram.size and n_peaks <= n_max_peaks):
        correction_to_global_max_index = Get_Corrected_Peak_Index(_histogram,
                                                                  peak_index,
                                                                  search_window_half_width)
        corrected_peak_index = int(peak_index + correction_to_global_max_index)
        
        if (_histogram[corrected_peak_index] > 0):
            # print(f"peak : {len(first_part_rows) + len(second_part_rows)} at position X={corrected_peak_index}")
            second_part_rows += [corrected_peak_index]
        
        peak_index = corrected_peak_index + _period
        n_peaks += 1    
    
    return first_part_rows[::-1]+second_part_rows[1:]

def Get_Signal_Period(_data, _axis_size, _bin_div):
    histogram = np.histogram(_data, bins=int(_axis_size/_bin_div), range=(0, _axis_size))
    power, freq = Compute_Power_and_Freq(histogram[0])
    signal_freq = Get_Signal_Freq(power, freq)
    signal_period = int(1/signal_freq)
    
    return histogram, signal_period

def All_Fourier_Analysis(image, max_rows=-1, max_cols=-1):      
    nrow, ncol = image.shape
    ################## Analyse signal on X axis
    X = np.sum(image, axis=1)
    X_pow, X_freq = Compute_Power_and_Freq(X)
    X_period = 1 / Get_Signal_Freq(X_pow, X_freq)
    crops_rows = np.sort(Search_Periodic_Peaks(X[:nrow], X_period))
    nb_rows = crops_rows.shape[0]

    print(f"found row period : {X_period}")
    print("nb_rows:", nb_rows)
            
    # ################## Analyse signal on Y axis
    Y = np.sum(image, axis=0)
    Y_pow, Y_freq = Compute_Power_and_Freq(Y)
    Y_period = 1 / Get_Signal_Freq(Y_pow, Y_freq)
    crops_cols = np.sort(Search_Periodic_Peaks(Y[:ncol], Y_period))
    nb_cols = crops_cols.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(X_pow)
    # ax[0].plot([X.mean() for i in range(X.shape[0])])
    ax[1].plot(Y_pow)
    # ax[1].plot([Y.mean() for i in range(Y.shape[0])])
    plt.show()

    print(f"found col period : {Y_period}")
    print("nb_cols:", nb_cols)

    print(crops_rows, crops_cols)

    initial_spot = np.array([np.min(crops_rows), np.min(crops_cols)])
    print(f"Intiial seed : {initial_spot}")

    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        # return arr.reshape(-1, la)
        return arr

    return initial_spot, cartesian_product(crops_rows, crops_cols)


def fit_fixed_size_grid(image, grid_shape=(12, 8), stride=(105, 105)):
    """
    Fit a grid of 12 rows and 8 columns.  Fixed stride of 105 pixels.
    """
    grid = np.zeros((grid_shape[0], grid_shape[1], 2))

    x0, y0 = 0, 0

    for i, line in enumerate(grid):
        for j, col in enumerate(line):
            grid[i, j] = [x0 + i * stride[0], y0 + j * stride[1]]

    np.save("D:\Documents\Thèse\Projet_Coli/image_analysis\Réplicat 1\grey/18a.npy", grid)  

# =============================================================================
# General Fourier Procedure
# =============================================================================

def pad_image(im, target_shape=(1500, 1500)):
    dx, dy = target_shape[0] - im.shape[0], target_shape[1] - im.shape[1]
    pad_x = np.zeros((dx, im.shape[1]))
    pad_y = np.zeros((target_shape[0], dy))
    im_with_defined_shape = np.concatenate([im, pad_x], axis=0)
    im_with_defined_shape = np.concatenate([im_with_defined_shape, pad_y], axis=1)
    return im_with_defined_shape

def find_starting_point(image, search_area=(200, 200), grid_size=(12, 8), grid_stride=(105, 105), patch_size=(20, 20), search_stride=25):

    def get_grid(i, j, imshape, gsize, gstride, psize):
        grid = np.zeros(imshape)
        nrowgrid, ncolgrid = gsize
        stride_x, stride_y = gstride
        p_x, p_y = psize
        for k in range(i, nrowgrid * stride_x, stride_x):
            for l in range(j, ncolgrid * stride_y, stride_y):
                t, b, l, r = np.max((0, k - p_x)), np.min((imshape[0], k + p_x)), np.max((0, l - p_y)), np.min((imshape[1], l + p_y))
                grid[t:b, l:r] = 1
        return grid

    scores = {}
    for i in range(0, search_area[0], search_stride):
        for j in range(0, search_area[1], search_stride):
            kernel = get_grid(i, j, image.shape, grid_size, grid_stride, patch_size)

            score = np.sum(image * kernel, axis=(0, 1))
            scores[(i, j)] = score

    return sorted(scores.items(), key=lambda x: x[1])[-1][0]

if (__name__ == "__main__"):
    # grid_size = (12, 8)
    # stride = (105, 105)
    # fit_fixed_size_grid(im, grid_size, stride)

    directory = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grey_farid"

    save_path = "D:\Documents\Thèse\Projet_Coli\image_analysis\Réplicat 1/3_starting_point.csv"

    grid_save_dir = "D:/Documents/Thèse/Projet_Coli/image_analysis/Réplicat 1/grids"
    if not os.path.isdir(grid_save_dir):
        os.mkdir(grid_save_dir)

    starting_points, stack = [], []
    for k, ifl in enumerate(sorted(os.listdir(directory))[:10]):
        extension = ifl.split(".")[-1]  # get the file extension

        if extension in ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]:
            if k % 10 == 0:
                dir_size = len(os.listdir(directory))
                print(f"Extracting image {k}/{dir_size}.")
            
            im = imageio.imread(directory + "/" + ifl) # go to grey scale for the preprocessing
            # im = difference_of_gaussians(im, low_sigma=2, high_sigma=4)
            # init_point, grid = All_Fourier_Analysis(np.abs(im - im.mean()), max_rows=12, max_cols=8)
            # starting_points.append([k, init_point[0], init_point[1]])

            x, y = find_starting_point(im)
            
            starting_points.append(np.array([k, x, y]))
            stack.append(pad_image(im))

            print("\n")

stack = np.array(stack)
starting_points = np.array(starting_points)

viewer = napari.view_image(stack)
napari.run()
