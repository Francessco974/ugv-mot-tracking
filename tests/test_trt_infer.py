import os
import pytest
from mot_tracking.detect.trt_infer import TRTInfer
import numpy as np

ENGINE_ENV_VAR = "TRT_ENGINE_PATH"
# Default WSL path using expanduser to resolve the ~ home directory
DEFAULT_WSL_ENGINE_PATH = os.path.expanduser("~/ugv_data/mot/models/yolov8n_mixed_pc.engine")


@pytest.fixture(scope="module")
def infer_engine():
    """Module-scoped fixture to load the engine once and handle missing files gracefully."""
    # Checks environment variable first; falls back to your WSL default path
    engine_path = os.environ.get(ENGINE_ENV_VAR, DEFAULT_WSL_ENGINE_PATH)
    if not os.path.exists(engine_path):
        pytest.skip(
            f"Engine file not found at '{engine_path}'. "
            f"Set the {ENGINE_ENV_VAR} environment variable or place the engine at the default path."
            )

    with TRTInfer(engine_path) as engine:
        yield engine


def test_output_shape_and_values(infer_engine):
    """Checks output tensor contract: shape (1, 84, 8400), float32, no NaN/Inf, valid scores."""
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    out = infer_engine.infer(x)

    assert out.shape == (1, 84, 8400), f"Expected shape (1, 84, 8400), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    assert not np.isnan(out).any(), "Output contains NaN values"
    assert not np.isinf(out).any(), "Output contains Inf values"

    # Class predictions are sigmoid outputs; bounded by [0.0, 1.0].
    # Adding a tight epsilon (1e-6) to account for single-precision float rounding in TRT FP16/INT8 math.
    classes = out[0, 4:, :]
    assert (classes >= 0.0).all() and (
        classes <= 1.0 + 1e-6
    ).all(), f"Class probabilities out of bounds [0, 1]. Max found: {classes.max()}"


def test_determinism(infer_engine):
    """Checks that identical inputs yield bitwise-identical outputs."""
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    out1 = infer_engine.infer(x)
    out2 = infer_engine.infer(x)

    np.testing.assert_array_equal(out1, out2)


def test_no_aliasing(infer_engine):
    """Ensures sequential calls do not mutate existing output references."""
    x1 = np.ones((1, 3, 640, 640), dtype=np.float32)
    x2 = np.zeros((1, 3, 640, 640), dtype=np.float32)

    out1 = infer_engine.infer(x1)
    out1_copy = out1.copy()

    # Second call reuses host buffers inside TRTInfer
    out2 = infer_engine.infer(x2)

    # out1 must remain unchanged
    np.testing.assert_array_equal(
        out1, out1_copy, err_msg="Output buffer was mutated across calls! Return copy missing."
    )
    assert not np.array_equal(out1, out2), "Expected different outputs for different inputs."


def test_reject_wrong_dtype(infer_engine):
    """Ensures non-float32 tensors are rejected."""
    x_f64 = np.zeros((1, 3, 640, 640), dtype=np.float64)
    with pytest.raises(TypeError):
        infer_engine.infer(x_f64)


def test_reject_wrong_shape(infer_engine):
    """Ensures tensors with incorrect dimensions are rejected."""
    x_wrong_shape = np.zeros((1, 3, 320, 320), dtype=np.float32)
    with pytest.raises(ValueError):
        infer_engine.infer(x_wrong_shape)


def test_reject_non_contiguous(infer_engine):
    """Ensures non-C-contiguous arrays with CORRECT shape are rejected strictly by contiguity check."""
    x_contiguous = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # Fortran order preserves shape (1, 3, 640, 640) but flips memory strides
    x_fortran = np.asfortranarray(x_contiguous)

    assert x_fortran.shape == infer_engine.input_shape, "Shape must match exactly for isolation"
    assert not x_fortran.flags["C_CONTIGUOUS"], "Test array must be non-contiguous"

    with pytest.raises(ValueError, match="C-contiguous"):
        infer_engine.infer(x_fortran)