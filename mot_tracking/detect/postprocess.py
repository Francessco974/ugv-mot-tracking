# mot_tracking/mot_tracking/detect/postprocess.py
import numpy as np
from mot_tracking.iou import compute_IoU
from mot_tracking.boxes import cxcywh_to_tlhw, tlwh_to_xyxy

   

def decode (detector_output : np.ndarray, conf_threshold: float = 0.05):
    """
    Decodes standard YOLO tensor of shape (1, 84, 8400).
    
    84 -> 4 (xc, yc, w, h) + 80 (class scores)
    class 0 -> person
    """
    
    # Remove batch dimension -> shape becomes (84, 8400)
    data = np.squeeze(detector_output)
    
    # Separate bounding box coordinates and class probabilities
    bounding_boxes = data[:4, :]  # Shape: (4, 8400)
    classes = data[4:, :]         # Shape: (80, 8400)
    
    # Identify highest scoring class and score for each anchor
    predicted_class = np.argmax(classes, axis=0)      # Shape: (8400,)
    max_confidence = np.max(classes, axis=0)           # Shape: (8400,)
    
    # Filter for class 0 ('person') above confidence threshold
    is_person = (predicted_class == 0) & (max_confidence >= conf_threshold)
    
    people_confidence = max_confidence[is_person]              # Shape: (N,)
    people_center_coord = bounding_boxes[:, is_person]         # Shape: (4, N)
    
    # Vectorized conversion from center (xc, yc, w, h) to top-left (left, top, w, h)
    if people_center_coord.shape[1] > 0: # if there are any people
        people_bb = cxcywh_to_tlhw(people_center_coord.T).T # change shape to (N,4)
    else:
        people_bb = np.empty((4, 0), dtype=float)
        
    return people_bb, people_confidence


def NMS(people_bb : np.ndarray, people_confidence : np.ndarray, IoU_threshold : float = 0.7, max_det : int =1000):
    """
    people_bb : Shape: (4, N)
    people_confidence : Shape: (N,)
    """
    
    if people_bb.size == 0 or people_bb.shape[1] == 0:
        return np.empty((0, 5), dtype=float)
    
    
    people_IoU = tlwh_to_xyxy(people_bb.T)
    result = []
    counter = 0
    while len(people_IoU) > 0:
        best_idx = np.argmax(people_confidence)
        
        confidence = people_confidence[best_idx]
        candidate = people_IoU[best_idx]
        
        iou = compute_IoU(people_IoU, np.reshape(candidate, (1, 4))).flatten()
        

        result.append(np.append((candidate), confidence))
        
        keep = iou < IoU_threshold
        people_IoU = people_IoU[keep]
        people_confidence = people_confidence[keep]
        counter += 1
        if counter >= max_det:
            break
        
    return np.array(result, dtype=float)
        

def clip_P1P2(NMS_output: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Clips bounding boxes in [x1, y1, x2, y2, confidence, ...] format to image boundaries
    and removes zero-area (collapsed) boxes.
    
    image_shape: (height, width)
    """
    if NMS_output.size == 0:
        return NMS_output

    height, width = image_shape
    data = NMS_output.copy()
    # Clip coordinates to image boundary
    data[:, 0] = np.clip(NMS_output[:, 0], 0.0, width)   # x1
    data[:, 1] = np.clip(NMS_output[:, 1], 0.0, height)  # y1
    data[:, 2] = np.clip(NMS_output[:, 2], 0.0, width)   # x2
    data[:, 3] = np.clip(NMS_output[:, 3], 0.0, height)  # y2
    
    # Keep only boxes where width > 0 AND height > 0
    mask = (data[:, 0] < data[:, 2]) & (data[:, 1] < data[:, 3])
    
    return data[mask]
    
    


def unletterbox_boxes(NMS_output: np.ndarray, scale: float, pad_w_l: int, pad_h_u: int) -> np.ndarray:
    """
    Maps bounding box coordinates [x1, y1, x2, y2, confidence, ...] back 
    from letterboxed space (e.g. 640x640) to original image space.
    """
    if NMS_output.size == 0:
        return NMS_output
        
    # Copy array to avoid mutating input in-place if needed
    boxes = NMS_output.copy()

    # 1. Subtract padding (Horizontal -> x1/x2, Vertical -> y1/y2)
    boxes[:, 0] -= pad_w_l  # x1
    boxes[:, 2] -= pad_w_l  # x2
    
    boxes[:, 1] -= pad_h_u  # y1
    boxes[:, 3] -= pad_h_u  # y2

    # 2. Divide by scaling factor
    boxes[:, 0:4] /= scale

    return boxes