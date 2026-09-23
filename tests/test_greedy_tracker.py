import numpy as np
from mot_tracking.tracker import greedy_match

def test_greedy_match_basic():
    iou_matrix = np.array([
        [0.1, 0.8],
        [0.7, 0.2]
    ])
    
    matches = greedy_match(iou_matrix, thr=0.5)

    assert len(matches) == 2
    assert (0, 1) in matches
    assert (1, 0) in matches

def test_greedy_match_below_threshold():
    iou_matrix = np.array([
        [0.2, 0.4],
        [0.1, 0.3]
    ])
    
    matches = greedy_match(iou_matrix, thr=0.5)
    assert len(matches) == 0