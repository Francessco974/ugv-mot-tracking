import pytest
import numpy as np
import cv2

from mot_tracking.detect.letterbox import (
    letter_box,
    from_BGR_to_RGB,
    from_HWC_to_CHW,
    from_int_to_float,
    transform_to_tensor,
    pre_process_image,
)
from mot_tracking.detect.postprocess import unletterbox_boxes

# -------------------------------------------------------------------------
# 1. Grounded Marker Test (Verifies letter_box output matches coordinate transforms)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("h, w", [
    (720, 1280), # Landscape (Pad Y)
    (1080, 720), # Portrait (Pad X)
    (320, 240),  # Upscaling
    (533, 811),  # Odd dimensions
])
def test_letterbox_marker_placement(h, w):
    """
    Places a solid white marker at a known position in the input image.
    Verifies that the physical location of pixels in the output image 
    matches the returned scale and padding.
    
    Marker size rationale (k = 11):
      - Large enough (11x11) so that interpolation and anti-aliasing do not decay
        the marker center intensity below threshold (> 200).
      - Small enough to isolate a precise local centroid.
      - Odd size yields an exact integer center index.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Place marker near top-left quadrant
    y0, x0 = h // 4, w // 4
    k = 11
    img[y0 : y0 + k, x0 : x0 + k] = 255
    
    # Ground truth marker centroid in original image space
    orig_center_x = x0 + (k - 1) / 2.0
    orig_center_y = y0 + (k - 1) / 2.0

    out, scale, (pad_w_l, pad_h_u) = letter_box(img, final_shape=(640, 640))

    # Find where the marker landed in the output image
    ys, xs = np.nonzero(out[..., 0] > 200)
    assert len(xs) > 0, "Marker was lost during letterbox resizing"

    actual_net_x = xs.mean()
    actual_net_y = ys.mean()

    # Theoretical placement in network space using returned scale and padding
    expected_net_x = (orig_center_x * scale) + pad_w_l
    expected_net_y = (orig_center_y * scale) + pad_h_u

    # Tolerance justification:
    # Combining integer pad rounding (<= 0.5px) and scale dimension rounding (<= 0.25px)
    # plus bilinear interpolation thresholding shift gives a max error bound of 1.0 px in 640x640 frame.
    np.testing.assert_allclose(
        [actual_net_x, actual_net_y],
        [expected_net_x, expected_net_y],
        atol=1.0
    )


# -------------------------------------------------------------------------
# 2. Derived Bound Tolerance for Image Center Mapping
# -------------------------------------------------------------------------

@pytest.mark.parametrize("orig_shape", [
    (720, 1280, 3), # Landscape
    (1080, 720, 3), # Portrait
    (320, 240, 3),  # Upscaling
    (533, 811, 3),  # Odd dimensions
])
def test_image_centre_derived_tolerance(orig_shape):
    """
    Verifies image center inverse mapping in original image frame, using a 
    rigorously derived bound: atol = 0.75 / scale.
    """
    h, w, _ = orig_shape
    dummy_img = np.zeros(orig_shape, dtype=np.uint8)
    _, scale, (pad_w_l, pad_h_u) = letter_box(dummy_img, final_shape=(640, 640))

    # Inverse mapping of network center (320, 320)
    orig_center_x = (320.0 - pad_w_l) / scale
    orig_center_y = (320.0 - pad_h_u) / scale

    expected_center = [w / 2.0, h / 2.0]
    
    # Derived max error in original pixels: 0.75 / scale
    derived_atol = 0.75 / scale

    np.testing.assert_allclose(
        [orig_center_x, orig_center_y],
        expected_center,
        atol=derived_atol
    )


# -------------------------------------------------------------------------
# 3. Partition Coverage: Upscaling Regime
# -------------------------------------------------------------------------

def test_upscaling_regime():
    """Verify scale > 1 when input is smaller than final shape."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    out, scale, (pad_w_l, pad_h_u) = letter_box(img, final_shape=(640, 640))

    expected_scale = 640.0 / 300.0
    assert np.isclose(scale, expected_scale)
    assert out.shape == (640, 640, 3)


# -------------------------------------------------------------------------
# 4. Color Channel Conversion Verification (BGR -> RGB)
# -------------------------------------------------------------------------

def test_bgr_to_rgb_conversion():
    """Verify BGR -> RGB correctly moves Blue from channel 0 to channel 2."""
    img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    img_bgr[:, :] = [255, 0, 0]  # Pure Blue in OpenCV BGR

    img_rgb = from_BGR_to_RGB(img_bgr)

    assert np.all(img_rgb[..., 0] == 0)    # Red channel
    assert np.all(img_rgb[..., 1] == 0)    # Green channel
    assert np.all(img_rgb[..., 2] == 255)  # Blue channel


# -------------------------------------------------------------------------
# 5. Layout & Spatial Permutation Verification (HWC -> CHW)
# -------------------------------------------------------------------------

def test_hwc_to_chw_layout_and_transpose():
    """
    Catches incorrect spatial/channel transpositions (e.g., transpose(2, 1, 0)).
    Uses asymmetric image dimensions and distinct per-pixel markers.
    """
    # Use asymmetric H and W to catch H/W permutation bugs
    img_hwc = np.zeros((10, 20, 3), dtype=np.uint8)
    
    # Set distinct pixel value at specific spatial location (y=2, x=5, c=1)
    img_hwc[2, 5, 1] = 180

    img_chw = from_HWC_to_CHW(img_hwc)

    assert img_chw.shape == (3, 10, 20)
    assert img_chw[1, 2, 5] == 180
    assert img_chw.flags['C_CONTIGUOUS']


# -------------------------------------------------------------------------
# 6. Pipeline Tensor Sanity Check
# -------------------------------------------------------------------------

def test_preprocess_image_pixel_value_mapping():
    """End-to-end check verifying that a pixel value traverses the pipeline correctly."""
    img_bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    
    # Use a 5x5 patch so bilinear interpolation maintains 1.0 intensity at the patch center
    y_idx, x_idx = 10, 20
    img_bgr[y_idx - 2 : y_idx + 3, x_idx - 2 : x_idx + 3] = [255, 0, 0]  # BGR Blue

    tensor, (scale, pad_w_l, pad_h_u) = pre_process_image(img_bgr)

    # Compute expected mapped center location in tensor
    expected_y = int(round(y_idx * scale + pad_h_u))
    expected_x = int(round(x_idx * scale + pad_w_l))

    # Tensor layout: (batch, channel, y, x). Blue in BGR becomes index 2 in RGB
    assert tensor[0, 2, expected_y, expected_x] > 0.9  # 255/255 = 1.0
    assert tensor[0, 0, expected_y, expected_x] == 0.0  # Red channel must be 0


# -------------------------------------------------------------------------
# 7. Invalid Input Handling
# -------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_img", [
    np.zeros((640, 640), dtype=np.uint8),     # 2D Grayscale
    np.zeros((640, 640, 4), dtype=np.uint8),  # 4 channels (RGBA)
])
def test_invalid_image_inputs(invalid_img):
    """Verify that improper channel dimensions raise ValueError."""
    with pytest.raises(ValueError):
        letter_box(invalid_img)
        
        
        
# -------------------------------------------------------------------------
# 8. G1.1 Round-Trip Transformation Test
# -------------------------------------------------------------------------

@pytest.mark.parametrize("camera_mode, orig_shape", [
    ("16:9 Landscape", (1080, 1920, 3)),
    ("4:3 Portrait/Standard", (640, 480, 3)),
])
def test_letterbox_roundtrip(camera_mode: str, orig_shape: tuple):
    """
    Test G1.1 Round-Trip Transformation:
    Original Point -> Letterbox Forward -> Unletterbox Inverse -> Assert Equality
    """
    h, w, _ = orig_shape
    dummy_img = np.zeros(orig_shape, dtype=np.uint8)

    # Invoke letter_box directly to obtain actual scale and padding values
    _, scale, (pad_w_l, pad_h_u) = letter_box(dummy_img, final_shape=(640, 640))

    # Define source bounding box coordinates in original image space
    x1_orig, y1_orig = w * 0.2, h * 0.25
    x2_orig, y2_orig = w * 0.6, h * 0.75
    confidence = 0.95

    # Apply forward letterbox coordinate transformation
    x1_lb = x1_orig * scale + pad_w_l
    y1_lb = y1_orig * scale + pad_h_u
    x2_lb = x2_orig * scale + pad_w_l
    y2_lb = y2_orig * scale + pad_h_u

    letterboxed_box = np.array([[x1_lb, y1_lb, x2_lb, y2_lb, confidence]], dtype=float)

    # Reconstruct original coordinates using inverse transformation
    restored_box = unletterbox_boxes(letterboxed_box, scale, pad_w_l, pad_h_u)
    expected_box = np.array([[x1_orig, y1_orig, x2_orig, y2_orig, confidence]], dtype=float)

    # Verify that restored coordinates match original values within tolerance
    np.testing.assert_allclose(restored_box, expected_box, rtol=1e-6, atol=1e-5)