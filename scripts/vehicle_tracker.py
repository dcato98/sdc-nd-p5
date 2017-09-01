# Standard imports
import cv2
import numpy as np
from scipy.ndimage.measurements import label

# Project imports
from project_utils import *

class VehicleTracker:
    """Tracks location of cars in a video"""
    
    def __init__(self, frame_shape, heat_gamma, heat_thresh):
        self.decay = heat_gamma
        self.heat_thresh = heat_thresh
        self.heatmap = np.zeros(frame_shape[0:2])
        self.out_frames = []
        
    def setup_object_finder(self, svc, X_scaler, y_start, y_stops, scales, color_space, 
                            orient, pix_per_cell, cell_per_block, spatial_size, n_hist_bins):
        self.svc = svc
        self.X_scaler = X_scaler
        self.y_start = y_start
        self.y_stops = y_stops
        self.scales = scales
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.n_hist_bins = n_hist_bins

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Find cars
        windows, confidences, = find_objects(frame, self.svc, self.X_scaler, self.y_start, self.y_stops, self.scales, self.color_space, 
                                             self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.n_hist_bins)
        
        # Find hot windows
        hot_windows = []
        for i, window in enumerate(windows):
            if confidences[i] > 0.5:
                hot_windows.append(window)

        # Update heatmap
        new_heat = add_heat(np.zeros_like(self.heatmap), hot_windows)
        # upper bound the pixelwise-heat gain per frame
        new_heat = np.clip(new_heat, 0.0, 2.5)
        # downweight old heat, add new heat, and blur (reduces jitteriness of car frames)
        self.heatmap *= self.decay
        self.heatmap += new_heat
        self.heatmap = cv2.GaussianBlur(self.heatmap, (3,3), 1)
        
        # Recognize cars as pixels where heat > threshold
        heat_frame = threshold(np.copy(self.heatmap), self.heat_thresh, 255)
        
        # Identify individual cars
        car_labels = label(heat_frame)
        
        # draw cars
        out_frame = BGR2('RGB', np.copy(frame))
        out_frame = draw_labeled_bboxes(out_frame, car_labels, color=(255,0,0))
        self.out_frames.append(out_frame)
        
        return out_frame