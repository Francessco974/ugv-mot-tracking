# tests/test_letterbox_roundtrip.py
import numpy as np
import pytest
from mot_tracking.detect.letterbox import letter_box
from mot_tracking.detect.postprocess import unletterbox_boxes


@pytest.fixture(
    params=[
        (720, 1280, "16:9_720p"),  # fov_mjpg_720p
        (480, 640, "4:3_480p"),    # fov_yuyv_480p
    ]
)
def mock_camera_frame(request):
    """
    Creates dummy images representing both camera modes:
    - 720p (720, 1280, 3)
    - 480p (480, 640, 3)
    """
    h, w, mode_name = request.param
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return img, h, w, mode_name


def forward_transform_points(pts: np.ndarray, scale: float, pad_w: float, pad_h: float) -> np.ndarray:
    """
    Applies forward letterbox transformation on points [x, y, ...]:
    x_lb = x * scale + pad_w
    y_lb = y * scale + pad_h
    """
    pts_lb = pts.copy().astype(float)
    pts_lb[..., 0] = pts_lb[..., 0] * scale + pad_w
    pts_lb[..., 1] = pts_lb[..., 1] * scale + pad_h
    return pts_lb


def test_letterbox_center_point_to_320_320(mock_camera_frame):
    """
    Verifies that the exact center of the original image maps forward
    to (320.0, 320.0) in the 640x640 letterboxed frame.
    """
    frame, orig_h, orig_w, _ = mock_camera_frame

    # Call letter_box without keyword argument 'new_shape'
    lb_img, scale, (pad_w, pad_h) = letter_box(frame, (640, 640))
    assert lb_img.shape == (640, 640, 3)

    # Image center in original coordinates
    orig_center_x = orig_w / 2.0
    orig_center_y = orig_h / 2.0
    orig_center = np.array([orig_center_x, orig_center_y], dtype=float)

    # Forward project to letterbox space
    lb_center = forward_transform_points(orig_center, scale, pad_w, pad_h)

    # Verify that original center lands exactly on (320, 320)
    expected_lb_center = np.array([320.0, 320.0], dtype=float)
    np.testing.assert_allclose(lb_center, expected_lb_center, rtol=1e-5, atol=1e-5)


def test_letterbox_roundtrip_bounding_boxes(mock_camera_frame):
    """
    Verifies full round-trip conversion:
    Original Box -> Letterbox Box -> Unletterbox Box == Original Box
    """
    frame, orig_h, orig_w, _ = mock_camera_frame

    # Letterbox image using positional target shape
    _, scale, (pad_w, pad_h) = letter_box(frame, (640, 640))

    # Define sample bounding boxes in original image space [x1, y1, x2, y2, score]
    orig_boxes = np.array([
        [0.0, 0.0, 100.0, 100.0, 0.9],                        # Top-left corner box
        [orig_w / 4.0, orig_h / 4.0, orig_w / 2.0, orig_h / 2.0, 0.85], # Mid box
        [orig_w - 50.0, orig_h - 50.0, orig_w, orig_h, 0.95]    # Bottom-right corner box
    ], dtype=float)

    # 1. Forward transform (Original -> Letterbox space)
    lb_boxes = orig_boxes.copy()
    lb_boxes[:, [0, 2]] = lb_boxes[:, [0, 2]] * scale + pad_w
    lb_boxes[:, [1, 3]] = lb_boxes[:, [1, 3]] * scale + pad_h

    # 2. Reverse transform using unletterbox_boxes (Letterbox -> Original space)
    recovered_boxes = unletterbox_boxes(lb_boxes, scale=scale, pad_w_l=pad_w, pad_h_u=pad_h)

    # 3. Assert recovery matches original boxes
    np.testing.assert_allclose(recovered_boxes, orig_boxes, rtol=1e-5, atol=1e-5)