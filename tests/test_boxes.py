import numpy as np
import pytest
from mot_tracking.boxes import (
    cxcywh_to_tlwh,
    tlwh_to_xyxy,
    xyxy_to_tlwh,
    tlwh_to_cxcysr,
    cxcysr_to_tlwh,
    xyxy_to_cxcysr,
    cxcysr_to_xyxy,
)

# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["1d", "2d_single", "2d_multiple"])
def sample_boxes(request):
    """
    Provides bounding boxes in different shapes:
    - 1D array: shape (4,)
    - 2D array with 1 box: shape (1, 4)
    - 2D array with N boxes: shape (N, 4)
    """
    mode = request.param
    tlwh_data = [
        [10.0, 20.0, 30.0, 40.0],
        [0.0, 0.0, 100.0, 200.0],
        [150.5, 250.25, 40.0, 80.0],
    ]

    if mode == "1d":
        return np.array(tlwh_data[0], dtype=float)
    elif mode == "2d_single":
        return np.array([tlwh_data[0]], dtype=float)
    else:  # 2d_multiple
        return np.array(tlwh_data, dtype=float)


# ---------------------------------------------------------------------------
# Round-Trip Tests (Invertibility Checks)
# ---------------------------------------------------------------------------

def test_roundtrip_tlwh_and_xyxy(sample_boxes):
    """Verifies that tlwh -> xyxy -> tlwh recovers original boxes."""
    xyxy = tlwh_to_xyxy(sample_boxes)
    recovered_tlwh = xyxy_to_tlwh(xyxy)
    np.testing.assert_allclose(recovered_tlwh, sample_boxes, rtol=1e-6, atol=1e-6)


def test_roundtrip_xyxy_and_tlwh(sample_boxes):
    """Verifies that xyxy -> tlwh -> xyxy recovers original boxes."""
    initial_xyxy = tlwh_to_xyxy(sample_boxes)
    
    tlwh = xyxy_to_tlwh(initial_xyxy)
    recovered_xyxy = tlwh_to_xyxy(tlwh)
    np.testing.assert_allclose(recovered_xyxy, initial_xyxy, rtol=1e-6, atol=1e-6)


def test_roundtrip_tlwh_and_cxcysr(sample_boxes):
    """Verifies that tlwh -> cxcysr -> tlwh recovers original boxes."""
    cxcysr = tlwh_to_cxcysr(sample_boxes)
    recovered_tlwh = cxcysr_to_tlwh(cxcysr)
    np.testing.assert_allclose(recovered_tlwh, sample_boxes, rtol=1e-6, atol=1e-6)


def test_roundtrip_cxcysr_and_tlwh(sample_boxes):
    """Verifies that cxcysr -> tlwh -> cxcysr recovers original boxes."""
    initial_cxcysr = tlwh_to_cxcysr(sample_boxes)
    
    tlwh = cxcysr_to_tlwh(initial_cxcysr)
    recovered_cxcysr = tlwh_to_cxcysr(tlwh)
    np.testing.assert_allclose(recovered_cxcysr, initial_cxcysr, rtol=1e-6, atol=1e-6)


def test_roundtrip_xyxy_and_cxcysr(sample_boxes):
    """Verifies that xyxy -> cxcysr -> xyxy recovers original boxes."""
    initial_xyxy = tlwh_to_xyxy(sample_boxes)
    
    cxcysr = xyxy_to_cxcysr(initial_xyxy)
    recovered_xyxy = cxcysr_to_xyxy(cxcysr)
    np.testing.assert_allclose(recovered_xyxy, initial_xyxy, rtol=1e-6, atol=1e-6)


def test_roundtrip_cxcysr_and_xyxy(sample_boxes):
    """Verifies that cxcysr -> xyxy -> cxcysr recovers original boxes."""
    initial_cxcysr = tlwh_to_cxcysr(sample_boxes)
    
    xyxy = cxcysr_to_xyxy(initial_cxcysr)
    recovered_cxcysr = xyxy_to_cxcysr(xyxy)
    np.testing.assert_allclose(recovered_cxcysr, initial_cxcysr, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Specific Value & Output Shape Verification
# ---------------------------------------------------------------------------

def test_cxcywh_to_tlwh_values():
    """Validates specific numerical calculations for cxcywh -> tlhw."""
    cxcywh = np.array([50.0, 60.0, 20.0, 40.0])  # [xc, yc, w, h]
    expected_tlhw = np.array([40.0, 40.0, 20.0, 40.0])  # [left, top, w, h]
    
    output = cxcywh_to_tlwh(cxcywh)
    np.testing.assert_allclose(output, expected_tlhw)


def test_tlwh_to_cxcysr_values():
    """Validates aspect ratio and area calculations in cxcysr format."""
    tlwh = np.array([10.0, 20.0, 30.0, 40.0])  # left, top, w, h
    # xc = 10 + 15 = 25, yc = 20 + 20 = 40, s = 30 * 40 = 1200, r = 30 / 40 = 0.75
    expected_cxcysr = np.array([25.0, 40.0, 1200.0, 0.75])
    
    output = tlwh_to_cxcysr(tlwh)
    np.testing.assert_allclose(output, expected_cxcysr)


def test_cxcysr_to_tlwh_values():
    """Validates conversion from cxcysr [25, 40, 1200, 0.75] to tlwh [10, 20, 30, 40]."""
    cxcysr = np.array([25.0, 40.0, 1200.0, 0.75])
    expected_tlwh = np.array([10.0, 20.0, 30.0, 40.0])

    output = cxcysr_to_tlwh(cxcysr)
    np.testing.assert_allclose(output, expected_tlwh)


def test_preserve_shapes():
    """Ensures input shape is strictly preserved (1D remains 1D, 2D remains 2D)."""
    single_1d = np.array([10.0, 20.0, 30.0, 40.0])
    batch_2d = np.array([[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 10.0, 10.0]])

    assert cxcywh_to_tlwh(single_1d).shape == (4,)
    assert cxcywh_to_tlwh(batch_2d).shape == (2, 4)

    assert tlwh_to_xyxy(single_1d).shape == (4,)
    assert tlwh_to_xyxy(batch_2d).shape == (2, 4)


# ---------------------------------------------------------------------------
# Contract, Empty Frames & Invalid Input Validation
# ---------------------------------------------------------------------------

def test_invalid_cxcysr_and_tlwh_raises():
    """Verifies that invalid or zero-dimension bounding boxes raise ValueError according to contract."""
    with pytest.raises(ValueError):
        tlwh_to_cxcysr(np.array([10.0, 20.0, 50.0, 0.0]))

    with pytest.raises(ValueError):
        cxcysr_to_tlwh(np.array([0.0, 0.0, -1.0, 1.0]))


@pytest.mark.parametrize(
    "func",
    [
        cxcywh_to_tlwh,
        tlwh_to_xyxy,
        xyxy_to_tlwh,
        tlwh_to_cxcysr,
        cxcysr_to_tlwh,
        xyxy_to_cxcysr,
        cxcysr_to_xyxy,
    ],
)
def test_empty_input_shape_preservation(func):
    """Verifies that passing an empty array of shape (0, 4) returns an empty array of shape (0, 4)."""
    empty_arr = np.empty((0, 4), dtype=float)
    result = func(empty_arr)
    assert result.shape == (0, 4)