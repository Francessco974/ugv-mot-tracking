import numpy as np
import pytest
from mot_tracking.iou import compute_IoU

def test_compute_iou_perfect_overlap():
    A = np.array([[10, 10, 50, 50]])
    B = np.array([[10, 10, 50, 50]])
    
    iou = compute_IoU(A, B)
    assert np.isclose(iou[0, 0], 1.0)

def test_compute_iou_no_overlap():
    A = np.array([[0, 0, 10, 10]])
    B = np.array([[20, 20, 30, 30]])
    
    iou = compute_IoU(A, B)
    assert np.isclose(iou[0, 0], 0.0)

def test_compute_iou_partial_overlap():
    A = np.array([[0, 0, 10, 10]])
    B = np.array([[5, 0, 15, 10]])
    
    iou = compute_IoU(A, B)
    assert np.isclose(iou[0, 0], 1/3)

def test_compute_iou_empty_inputs():
    A = np.empty((0, 4))
    B = np.array([[10, 10, 50, 50]])
    
    iou = compute_IoU(A, B)
    assert iou.shape == (0, 1)