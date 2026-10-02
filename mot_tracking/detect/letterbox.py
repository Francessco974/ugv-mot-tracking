# mot_tracking/mot_tracking/detect/letterbox.py
import numpy as np
import cv2

def letter_box(image : np.ndarray, final_shape = (640, 640)):
    # input shape = (Height, Width, Channels)
    # image centre maps to H/2, W/2
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image input must have HWC shape")
    
    target_height, target_width = final_shape  # assuming standard (H, W)
    
    height, width, _ = image.shape
    
    scale = min (target_height/height, target_width /width )
    
    scaled_width  = round(width*scale)
    scaled_height = round(height*scale)
    
    resized_image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR) 
    
    pad_w = target_width  - scaled_width
    pad_h = target_height - scaled_height
    
    pad_w_l = int(np.floor (pad_w/2))
    pad_w_r = int(np.ceil  (pad_w/2))
    pad_h_u = int(np.floor (pad_h/2))
    pad_h_b = int(np.ceil  (pad_h/2))
    
    letterboxed_image = cv2.copyMakeBorder(
        resized_image,
        top=pad_h_u,
        bottom=pad_h_b,
        left=pad_w_l,
        right=pad_w_r,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114)  # Gray border
    )
    
    # Return image along with parameters required for coordinate mapping
    return letterboxed_image, scale, (pad_w_l, pad_h_u)


def from_BGR_to_RGB (image_bgr : np.ndarray):
    """input shape = (Height, Width, Channels)"""
    
    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Image input must have HWC shape")
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def from_HWC_to_CHW (image_hwc : np.ndarray):
    """input shape = (Height, Width, Channels)"""
    
    if len(image_hwc.shape) != 3 or image_hwc.shape[2] != 3:
        raise ValueError("Image input must have HWC shape")
    
    return np.ascontiguousarray(image_hwc.transpose(2, 0, 1))

def from_int_to_float(image_int: np.ndarray) -> np.ndarray:
    if image_int.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {image_int.dtype}")
    return image_int.astype(np.float32) / 255.0

def transform_to_tensor(image : np.ndarray):
    return np.expand_dims(image, axis=0)


def pre_process_image(image):
    """input shape = (Height, Width, Channels)"""
    letterboxed_image, scale, (pad_w_l, pad_h_u) = letter_box(image)
    pre_procesed_image = from_int_to_float(from_HWC_to_CHW(from_BGR_to_RGB(letterboxed_image)))
    return transform_to_tensor(pre_procesed_image), (scale, pad_w_l, pad_h_u)

