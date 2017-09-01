import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from skimage.feature import hog

def find_files_generator(root_dir, extensions=()):
    """Searches for and yields files in directory and subdirectories"""
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extensions) or extensions == ():
                yield os.path.join(root, file)

def get_hog_features(image, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    """
    Calculates HOG features and (optionally) HOG visualization.
    
    See scikit-learn documentation for an explanation of HOG features here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog
    
    Returns:
        `features` if `vis` is False
        `(features, hog_image)` if `vis` is True
    """   
    hog_output = hog(image, orientations=orient,
                     pixels_per_cell=(pix_per_cell, pix_per_cell),
                     cells_per_block=(cell_per_block, cell_per_block),
                     block_norm='L2-Hys', transform_sqrt=True,
                     visualise=vis, feature_vector=feature_vec)
    return hog_output

def bin_spatial(img, size=(32, 32)):
    """Returns a vector of binned color features"""
    spatial_features = cv2.resize(img, size).ravel()
    return spatial_features

def color_hist(img, n_bins=32, bins_range=(0, 256)):
    """Returns color histogram features"""
    # Compute the histogram of each color channel
    channel1_hist = np.histogram(img[:,:,0], bins=n_bins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=n_bins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=n_bins, range=bins_range)
    # Concatenate into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def BGR2(color_space, image):
    """Converts BGR image to another color space."""
    color_space = color_space.upper()
    if color_space == 'BGR':
        return image.copy()
    elif color_space == 'GRAY': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'RGB': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LUV': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif color_space == 'HLS': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif color_space == 'YUV': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif color_space == 'YCRCB': converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError("Invalid Argument: 'color_space' must be one of: 'BGR', 'gray', 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'")
    return converted_image

def extract_features(images, color_space='BGR', spatial_size=(32, 32), n_hist_bins=32, orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extracts and concatenates a vector of HOG, color, and spatial features for each image"""
    
    # Initialize list for collecting each image's feature vector
    features = []
    
    # Extract feature vector from each image
    for image in images:
        
        # Initialize list for collecting image features
        feature_vector = []
        
        # Apply color conversion
        feature_image = BGR2(color_space, image)

        # Collect spatial features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            feature_vector.append(spatial_features)
            
        # Collect color features
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, n_bins=n_hist_bins)
            feature_vector.append(hist_features)
            
        # Collect HOG features
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_feat1 = (get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block))
                hog_feat2 = (get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block))
                hog_feat3 = (get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block))
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block)
            feature_vector.append(hog_features)
        
        # Append this image's feature vector to the list
        features.append(np.concatenate(feature_vector))
        
    # Return list of feature vectors
    return features

def draw_boxes(image, bboxes, color=(255, 0, 0), thick=6):
    """Draws bounding boxes on `image`"""
    for bbox in bboxes:
        cv2.rectangle(image, bbox[0], bbox[1], color=color, thickness=thick)
    return image

def pixel_plot3d(pixels, colors_bgr, axis_labels=list("BGR"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_bgr
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_bgr.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

def add_heat(heatmap, bbox_list, values=None):
    """Add heat to heatmap at bounding box locations"""
    if not values:
        values = [1.0]*len(bbox_list)
    for i, box in enumerate(bbox_list):
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += values[i]
    return heatmap
    
def threshold(array, min_val=None, max_val=None, update_val=0.0):
    """Sets elements outside of the threshold to `value`"""
    if min_val and max_val:
        array[(array < min_val) | (array > max_val)] = update_val
    elif min_val:
        array[array < min_val] = update_val
    elif max_val:
        array[array > max_val] = update_val
    return array

def bboxes_from_labels(labels):
    """Converts labels (2d binary images) to bounding boxes"""
    bboxes = []
    for label in range(1, labels[1]+1):
        # Find pixels with each label value
        y_nonzero, x_nonzero = (labels[0] == label).nonzero()
        # Define bounding box based on min/max x and y
        bbox = ((np.min(x_nonzero), np.min(y_nonzero)), (np.max(x_nonzero), np.max(y_nonzero)))
        bboxes.append(bbox)
    return bboxes

def draw_labeled_bboxes(image, labels, color=(255, 0, 0), thick=6):
    """Draw labeled bounding boxes on image"""
    bboxes = bboxes_from_labels(labels)
    for bbox in bboxes:
        cv2.rectangle(image, bbox[0], bbox[1], color, thick)
    return image

def find_objects(image, object_svc, X_scaler, y_start, y_stops, scales=[1.5], color_space='YCrCb',
                 orient=9, pix_per_cell=8, cells_per_block=2, spatial_size=(32,32), n_hist_bins=32):
    """
    Finds objects in images by:
     1) breaking up images into patches
     2) computing HOG, color, and histogram features
     3) generating a prediction using a object classifier
     
     Note: Uses HOG sub-sampling for efficiency.
    
    Parameters:
        image - the image to search
        object_svc - classifier which predicts objects from HOG, color, and histogram features
        X_scaler - scales features for prediction by the classifier
        y_start - lower y-bound on image search location
        y_stop - upper y-bound on image search location
        scales - list of scales, increase/decrease scale to search larger/smaller sized image patches 
                 (scale=1 searches 64x64 image patches)
        orient - for HOG features: number of discrete orientations
        pix_per_cell - for HOG features: number of pixels per cell
        cells_per_block - for HOG features: number of cells per block
        spatial_size - for color features: size of the feature
        hist_bins - for histogram features: number of histogram bins
        draw_color - color of object bounding box
        draw_thickness - thickness object bounding box
    
    Returns a tuple containing:
        windows - a list of all bounding boxes for which the classifier made a positive prediction
        confidences - a list of corresponding confidence values, confidence > 0 is a positive prediction
    """
    windows = []
    confidences = []
    
    for i, scale in enumerate(scales):
        y_stop = y_stops[i]
        
        # Initialize, recolor, and rescale image for computing features
        search_img = image[y_start:y_stop,:,:]
        search_color = BGR2(color_space, search_img)
        if scale != 1:
            search_color = cv2.resize(search_color, (np.int(search_color.shape[1]/scale), np.int(search_color.shape[0]/scale)))
        imshape = search_color.shape

        # Define blocks and steps
        features_per_block = orient*cells_per_block**2
        cells_per_step = 1
        pix_per_window = 64
        blocks_per_window = (pix_per_window // pix_per_cell) - cells_per_block + 1
        n_xy_blocks = (np.array([imshape[1], imshape[0]]) // pix_per_cell) - cells_per_block + 1
        n_xy_steps = (n_xy_blocks - blocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        ch1, ch2, ch3 = search_color[:,:,0], search_color[:,:,1], search_color[:,:,2]
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cells_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cells_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cells_per_block, vis=False, feature_vec=False)

        # Extract features and make prediction for each window
        for x_win in range(n_xy_steps[0]):
            for y_win in range(n_xy_steps[1]):
                x_pos = x_win*cells_per_step
                y_pos = y_win*cells_per_step
                x_left = x_pos*pix_per_cell
                y_top = y_pos*pix_per_cell

                # Extract HOG for this patch
                hog_feat1 = hog1[y_pos:y_pos+blocks_per_window, x_pos:x_pos+blocks_per_window].ravel() 
                hog_feat2 = hog2[y_pos:y_pos+blocks_per_window, x_pos:x_pos+blocks_per_window].ravel() 
                hog_feat3 = hog3[y_pos:y_pos+blocks_per_window, x_pos:x_pos+blocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                # Extract the image patch
                patch = search_color[y_top:y_top+pix_per_window, x_left:x_left+pix_per_window]

                # Get color features
                spatial_features = bin_spatial(patch, size=spatial_size)
                hist_features = color_hist(patch, n_bins=n_hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_confidence = object_svc.decision_function(test_features)
                
                x_win_left = np.int(x_left*scale)
                y_win_top = np.int(y_top*scale)+y_start
                x_win_right = x_win_left + np.int(pix_per_window*scale)
                y_win_bottom = y_win_top + np.int(pix_per_window*scale)
                window = ((x_win_left, y_win_top), (x_win_right, y_win_bottom))
                windows.append(window)
                confidences.append(test_confidence)
    
    return (windows, confidences)