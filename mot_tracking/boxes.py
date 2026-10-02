# mot_tracking/mot_tracking/boxes.py
import numpy as np


def cxcywh_to_tlwh(cxcywh : np.ndarray) -> np.ndarray:
    """
    Converts array of shape (N, 4) or (4,) from [xc, yc, w, h] to [left, top, w, h].
    """
    
    cxcywh = np.asarray(cxcywh, dtype=float)
    
    
    # Extract coordinates along the last axis (-1) to work for both 1D and 2D
    xc, yc, w, h = cxcywh[..., 0], cxcywh[..., 1], cxcywh[..., 2], cxcywh[..., 3]
    
    
    left = xc - w / 2.0
    top = yc - h / 2.0
    
    return np.stack([left, top, w, h], axis=-1)
    
def tlwh_to_xyxy(tlwh : np.ndarray) -> np.ndarray: 
    """
    Converts array of shape (N, 4) or (4,) from [left, top, w, h] to [x1, y1, x2, y2].
    """
    
    tlwh = np.asarray(tlwh, dtype=float)
    
    left = tlwh[..., 0]
    up = tlwh[..., 1]
    width = tlwh[..., 2]
    height = tlwh[..., 3]
    
    right = left + width
    down = up + height
    
    return np.stack([left, up, right, down], axis=-1)

def xyxy_to_tlwh(xyxy : np.ndarray) -> np.ndarray: 
    """
    Converts array of shape (N, 4) or (4,) from [x1, y1, x2, y2] to [left, top, w, h].
    """
    
    xyxy = np.asarray(xyxy, dtype=float)
    
    x1 = xyxy[..., 0]
    y1 = xyxy[..., 1]
    x2 = xyxy[..., 2]
    y2 = xyxy[..., 3]
    
    w = x2 - x1
    h = y2 - y1
    
    return np.stack([x1, y1, w, h], axis=-1)

def tlwh_to_cxcysr(tlwh : np.ndarray) -> np.ndarray:
    """
    Converts array of shape (N, 4) or (4,) from [left, top, w, h] to [xc yc s_(area) r_(aspect ratio)].
    """
    
    tlwh = np.asarray(tlwh, dtype=float)
    
    left = tlwh[..., 0]
    up = tlwh[..., 1]
    width = tlwh[..., 2]
    height = tlwh[..., 3]
    
    xc = left + width / 2.0
    yc = up + height / 2.0
    s =  width * height
    
    if np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("Degenerate box: should have been removed upstream")
    r = width / height

    return np.stack([xc, yc, s, r], axis=-1)

def cxcysr_to_tlwh(cxcysr : np.ndarray) -> np.ndarray:
    """
    Converts array of shape (N, 4) or (4,) from [xc yc s_(area) r_(aspect ratio)] to [left, top, w, h].
    """
    
    cxcysr = np.asarray(cxcysr, dtype=float)
    
    cx = cxcysr[..., 0]
    cy = cxcysr[..., 1]
    s = cxcysr[..., 2]
    r = cxcysr[..., 3]
    
    if np.any(s < 0) or np.any(r <= 0):
        raise ValueError("Invalid state: s < 0 or r <= 0")
        
    w = np.sqrt(s * r)
    h = np.sqrt(s / r)

    left = cx - w / 2.0
    top   = cy - h / 2.0

    return np.stack([left, top, w, h], axis=-1)

def xyxy_to_cxcysr(xyxy : np.ndarray) -> np.ndarray:
    """
    Converts array of shape (N, 4) or (4,) from [x1, y1, x2, y2] to [xc yc s_(area) r_(aspect ratio)].
    """
    return  tlwh_to_cxcysr(tlwh=xyxy_to_tlwh(xyxy))

def cxcysr_to_xyxy(cxcysr : np.ndarray) -> np.ndarray:
    """
    Converts array of shape (N, 4) or (4,) from [xc yc s_(area) r_(aspect ratio)] to [x1, y1, x2, y2].
    """
    return  tlwh_to_xyxy(tlwh=cxcysr_to_tlwh(cxcysr))

def cxcywh_to_xyxy(cxcywh):
    """
    Converts array of shape (N, 4) or (4,) from [xc, yc, w, h] to [x1, y1, x2, y2].
    """
    return tlwh_to_xyxy(cxcywh_to_tlwh(cxcywh))






# from mot_tracking.boxes import cxcywh_to_tlwh, tlwh_to_xyxy, xyxy_to_tlwh, tlwh_to_cxcysr, cxcysr_to_tlwh, xyxy_to_cxcysr, cxcysr_to_xyxy