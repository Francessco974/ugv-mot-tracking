import numpy as np
import pytest
from mot_tracking.tracker import greedy_match, hungarian_match

iou = np.array([[0.9, 0.85],
                 [0.8, 0.1]])


def test_greedy():
    matches = greedy_match(iou, 0.05)
    assert (matches[0][0] == 0)
    assert (matches[0][1] == 0)
    assert (matches[1][0] == 1)
    assert (matches[1][1] == 1)
    
def test_hungarian():
    matches = hungarian_match(iou, 0.05)
    assert (matches[0][0] == 0)
    assert (matches[0][1] == 1)
    assert (matches[1][0] == 1)
    assert (matches[1][1] == 0)