# mot_tracking/iou.py
import numpy as np

def to_IoU_coordinates(bb_MOT_coordinates):
    # [left, up, width, height]
    left, up, width, height = bb_MOT_coordinates
    return [left, up, left+width, up+height]

def to_BB_coordinates(bb_IoU_coordinates):
    # [x1 , y1 , x2, y2]
    x1 , y1 , x2, y2 = bb_IoU_coordinates
    return [x1, y1, x2-x1, y2-y1]



def rectangle_area(corners):
    return max((corners[2]- corners[0]), 0.0) * max((corners[3]- corners[1]), 0.0)


def compute_IoU (A, B):
    
    # Expand dims for broadcasting: (N, 1, 4) and (1, M, 4)
    A_exp = A[:, np.newaxis, :]  # Shape: (N, 1, 4)
    B_exp = B[np.newaxis, :, :]  # Shape: (1, M, 4)
    
    # top-left intersection corner x1, y1
    max_first_two = np.maximum(A_exp[:, :, :2], B_exp[:, :, :2])

    # bottom-right intersection corner x2, y2
    min_last_two = np.minimum(A_exp[:, :, 2:], B_exp[:, :, 2:])

    # Concatenate back along the last dimension (axis=2)
    intersection_corners=  np.concatenate([max_first_two, min_last_two], axis=2)
        
    # Vectorized way to compute Areas
    inter_widths = np.maximum(0.0, intersection_corners[:, :, 2] - intersection_corners[:, :, 0])
    inter_heights = np.maximum(0.0, intersection_corners[:, :, 3] - intersection_corners[:, :, 1])
    
    # Intersection Areas: (N, M)
    inter_areas = inter_widths * inter_heights
    
    # A
    A_widths = np.maximum(0.0, A_exp[:, :, 2] - A_exp[:, :, 0])
    A_heights = np.maximum(0.0, A_exp[:, :, 3] - A_exp[:, :, 1])
    A_areas = A_widths * A_heights
    
    # B
    B_widths = np.maximum(0.0, B_exp[:, :, 2] - B_exp[:, :, 0])
    B_heights = np.maximum(0.0, B_exp[:, :, 3] - B_exp[:, :, 1])
    B_areas = B_widths * B_heights
    
    # Union Areas: (N, M)
    union_areas = A_areas + B_areas - inter_areas
    
    # Avoid division by zero when union_areas is 0
    return np.where(union_areas > 0, inter_areas / union_areas, 0.0)
    
    
    
    
    
    
        
        
    