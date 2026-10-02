# mot_tracking/iou.py
import numpy as np


def compute_IoU (A, B):
    if A.ndim != 2 or A.shape[1] != 4: raise ValueError(f"A must be (N, 4), got {A.shape}")
    if B.ndim != 2 or B.shape[1] != 4: raise ValueError(f"B must be (N, 4), got {B.shape}")
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
    
    
    out = np.zeros_like(inter_areas)
    np.divide(
        inter_areas,
        union_areas,
        out=out,
        where=union_areas > 0
    )

    return out
    
    
    
    
    
        
        
    