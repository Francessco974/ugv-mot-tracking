# mot_tracking/mot_tracking/detect/trt_infer.py
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def check_cuda(err: cudart.cudaError_t) -> None:
    """Helper to check CUDA runtime call return status and raise on error."""
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error: {err}")


class TRTInfer:
    """TensorRT Inference Engine wrapper with explicit CUDA memory management."""

    def __init__(self, engine_path: str):
        # 1. Logger + runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # 2. Deserialize engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load engine from {engine_path}. "
                "The engine may have been built for a different TensorRT version or GPU architecture."
            )

        # 3. Create execution context
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # 4. Discover I/O Tensors
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.output_dtype = None

        num_tensors = self.engine.num_io_tensors
        for i in range(num_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                if self.input_name is not None:
                    raise ValueError("Multiple input tensors found. Expected exactly 1.")
                self.input_name = name
                self.input_shape = shape
                self.input_dtype = dtype
            elif mode == trt.TensorIOMode.OUTPUT:
                if self.output_name is not None:
                    raise ValueError("Multiple output tensors found. Expected exactly 1.")
                self.output_name = name
                self.output_shape = shape
                self.output_dtype = dtype

        # 5. Validate model specifications
        if self.input_name is None or self.output_name is None:
            raise ValueError("Engine must have exactly 1 input and 1 output tensor.")

        for shape, label in [(self.input_shape, "Input"), (self.output_shape, "Output")]:
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"{label} shape {shape} contains dynamic/unbound dimensions (-1). Static shapes required.")

        if self.input_dtype != np.float32 or self.output_dtype != np.float32:
            raise TypeError(
                f"Data type mismatch: Expected float32 input/output, "
                f"got input={self.input_dtype}, output={self.output_dtype}"
            )

        # 6. Allocate GPU memory and Host buffer
        input_nbytes = int(np.prod(self.input_shape)) * np.dtype(self.input_dtype).itemsize
        output_nbytes = int(np.prod(self.output_shape)) * np.dtype(self.output_dtype).itemsize

        err, self.d_input = cudart.cudaMalloc(input_nbytes)
        check_cuda(err)

        err, self.d_output = cudart.cudaMalloc(output_nbytes)
        check_cuda(err)

        self.h_output = np.zeros(self.output_shape, dtype=self.output_dtype)

        # 7. Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        check_cuda(err)

        # 8. Bind addresses (Once, as buffers don't move)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Runs inference on a preprocessed C-contiguous numpy tensor."""
        # 1. Validate Input
        if x.shape != self.input_shape:
            raise ValueError(f"Input shape mismatch: expected {self.input_shape}, got {x.shape}")
        if x.dtype != self.input_dtype:
            raise TypeError(f"Input dtype mismatch: expected {self.input_dtype}, got {x.dtype}")
        if not x.flags["C_CONTIGUOUS"]:
            raise ValueError("Input array must be C-contiguous to prevent scrambled GPU memory copies.")

        # 2. Host-to-Device (H2D) copy async
        err, = cudart.cudaMemcpyAsync(
            self.d_input,
            x.ctypes.data,
            x.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        check_cuda(err)

        # 3. Enqueue TensorRT execution
        success = self.context.execute_async_v3(int(self.stream))
        if not success:
            raise RuntimeError("TensorRT execute_async_v3 failed to enqueue kernel.")

        # 4. Device-to-Host (D2H) copy async
        err, = cudart.cudaMemcpyAsync(
            self.h_output.ctypes.data,
            self.d_output,
            self.h_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        check_cuda(err)

        # 5. Synchronize Stream
        err, = cudart.cudaStreamSynchronize(self.stream)
        check_cuda(err)

        # 6. Return deep copy to eliminate array aliasing across calls
        return self.h_output.copy()

    def close(self) -> None:
        """Frees all allocated GPU pointers and CUDA stream handles."""
        if getattr(self, "d_input", None) is not None:
            check_cuda(cudart.cudaFree(self.d_input)[0])
            self.d_input = None

        if getattr(self, "d_output", None) is not None:
            check_cuda(cudart.cudaFree(self.d_output)[0])
            self.d_output = None

        if getattr(self, "stream", None) is not None:
            check_cuda(cudart.cudaStreamDestroy(self.stream)[0])
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        # Best-effort cleanup upon garbage collection
        try:
            self.close()
        except Exception:
            pass