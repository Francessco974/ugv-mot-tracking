import pytest
import numpy as np

from mot_tracking.detect.letterbox import letter_box
from mot_tracking.detect.postprocess import (
    unletterbox_boxes,
    clip_P1P2,
    decode,
    from_center_to_bb,
    from_bb_to_P1_P2,
)

# -------------------------------------------------------------------------
# 1. G1.1 Round-Trip Transformation Test
# -------------------------------------------------------------------------

@pytest.mark.parametrize("camera_mode, orig_shape", [
    ("16:9 Landscape", (1080, 1920, 3)),
    ("4:3 Portrait", (640, 480, 3)),
])
def test_G1_1_round_trip_transformation(camera_mode, orig_shape):
    """
    G1.1 Round-Trip Test:
    1. Take a ground-truth pixel coordinate in original image space: (x, y).
    2. Apply letterbox forward transformation: x_lb = x * scale + pad_w_l
    3. Pass through unletterbox_boxes to reverse the transformation.
    4. Assert that the recovered point equals the starting point using np.allclose.
    """
    h, w, _ = orig_shape
    dummy_img = np.zeros(orig_shape, dtype=np.uint8)
    
    # Obtain exact scale and padding produced by letter_box
    _, scale, (pad_w_l, pad_h_u) = letter_box(dummy_img, final_shape=(640, 640))

    # Define test point corners in original image space [x1, y1, x2, y2, confidence]
    x1_orig, y1_orig = w * 0.2, h * 0.25
    x2_orig, y2_orig = w * 0.6, h * 0.75
    confidence = 0.88

    # Apply forward letterbox transformation (x * scale + pad)
    x1_lb = x1_orig * scale + pad_w_l
    y1_lb = y1_orig * scale + pad_h_u
    x2_lb = x2_orig * scale + pad_w_l
    y2_lb = y2_orig * scale + pad_h_u

    letterboxed_box = np.array([[x1_lb, y1_lb, x2_lb, y2_lb, confidence]], dtype=float)

    # Reconstruct original box using unletterbox_boxes
    restored_box = unletterbox_boxes(letterboxed_box, scale, pad_w_l, pad_h_u)
    expected_box = np.array([[x1_orig, y1_orig, x2_orig, y2_orig, confidence]], dtype=float)

    # Verify perfect round-trip reconstruction
    np.testing.assert_allclose(
        restored_box,
        expected_box,
        rtol=1e-6,
        atol=1e-5,
        err_msg=f"G1.1 Round-trip transformation failed for camera mode: {camera_mode}"
    )


# -------------------------------------------------------------------------
# 2. Direct Tests for unletterbox_boxes
# -------------------------------------------------------------------------

def test_unletterbox_boxes_empty_input():
    """Verify gracefully handling empty detection arrays."""
    empty_boxes = np.empty((0, 5), dtype=float)
    res = unletterbox_boxes(empty_boxes, scale=0.5, pad_w_l=32, pad_h_u=0)
    assert res.size == 0


def test_unletterbox_boxes_immutability():
    """Verify unletterbox_boxes returns a copy and does not mutate the input array."""
    input_box = np.array([[100.0, 50.0, 200.0, 150.0, 0.9]], dtype=float)
    original_copy = input_box.copy()
    
    _ = unletterbox_boxes(input_box, scale=0.5, pad_w_l=10, pad_h_u=10)
    
    np.testing.assert_array_equal(input_box, original_copy)


# -------------------------------------------------------------------------
# 3. Direct Tests for clip_P1P2
# -------------------------------------------------------------------------

def test_clip_P1P2_clipping_and_filtering():
    """Verify out-of-bounds clipping and zero/negative area box removal."""
    image_shape = (480, 640)  # height, width
    
    boxes = np.array([
        [-10.0, -20.0, 700.0, 500.0, 0.95],  # Out of bounds -> clipped to [0, 0, 640, 480]
        [100.0, 100.0, 100.0, 200.0, 0.80],  # Zero width (x1 == x2) -> should be removed
        [300.0, 200.0, 200.0, 400.0, 0.70],  # Negative width (x1 > x2) -> should be removed
        [50.0, 50.0, 150.0, 150.0, 0.90],    # Valid box -> kept unchanged
    ], dtype=float)

    clipped = clip_P1P2(boxes, image_shape=image_shape)

    # Expect exactly 2 boxes remaining: box 0 (clipped) and box 3 (valid)
    assert len(clipped) == 2
    
    # Check clipped box 0 coordinates
    expected_box_0 = np.array([0.0, 0.0, 640.0, 480.0, 0.95])
    np.testing.assert_allclose(clipped[0], expected_box_0)


# -------------------------------------------------------------------------
# 4. Direct Tests for Decoding Functions
# -------------------------------------------------------------------------

def test_center_to_bb_conversion():
    """Verify conversion from [xc, yc, w, h] to [left, top, w, h]."""
    # 1D array shape (4,) matching standard bounding box vectors
    center_coords = np.array([100.0, 200.0, 40.0, 60.0], dtype=float)
    expected_bb = np.array([80.0, 170.0, 40.0, 60.0], dtype=float)

    res = from_center_to_bb(center_coords)
    np.testing.assert_allclose(res.flatten(), expected_bb)


def test_decode_yolo_tensor():
    """Verify decoding YOLO tensor for person class thresholding."""
    dummy_detector_out = np.zeros((1, 84, 8400), dtype=float)

    # Setup anchor 0: xc=100, yc=100, w=20, h=40 -> left = 100 - 10 = 90, top = 100 - 20 = 80
    dummy_detector_out[0, 0:4, 0] = [100.0, 100.0, 20.0, 40.0]
    dummy_detector_out[0, 4, 0] = 0.80  # Class 0 score (person)

    # Setup anchor 1: non-person class score (index 5)
    dummy_detector_out[0, 0:4, 1] = [200.0, 200.0, 30.0, 30.0]
    dummy_detector_out[0, 5, 1] = 0.90

    boxes, confs = decode(dummy_detector_out, conf_threshold=0.5)

    assert len(confs) == 1
    assert confs[0] == 0.80
    # left = 90.0, top = 80.0, w = 20.0, h = 40.0
    np.testing.assert_allclose(boxes[:, 0], [90.0, 80.0, 20.0, 40.0])