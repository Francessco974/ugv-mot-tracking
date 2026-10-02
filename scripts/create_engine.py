#!/usr/bin/env python3
"""Build a strongly typed TensorRT engine from an ONNX file.

Precision comes from the ONNX graph itself (no FP16 builder flag):
use the AutoCast'd model (yolov8n_mixed.onnx) for mixed FP16,
the original model for an FP32 reference engine.

Works on TensorRT 11 (strongly typed by default) and 10.12+ (flag required).

Usage:
    python create_engine.py models/yolov8n_mixed.onnx models/yolov8n_mixed_pc.engine
    python create_engine.py models/yolov8n.onnx       models/yolov8n_fp32_pc.engine
"""
import argparse
import os

import tensorrt as trt


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX -> strongly typed TensorRT engine")
    ap.add_argument("onnx", help="input ONNX path")
    ap.add_argument("engine", help="output engine path")
    args = ap.parse_args()

    onnx_path = os.path.expanduser(args.onnx)
    engine_path = os.path.expanduser(args.engine)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print(f"TensorRT {trt.__version__}")

    # TRT 11: every network is strongly typed, no flag needed.
    # TRT 10.12+: STRONGLY_TYPED must be requested, otherwise the build is weakly typed.
    major = int(trt.__version__.split(".")[0])
    flags = 0
    if major < 11:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    # I/O must stay float32: TRTInfer, pre-processing and decode all expect it.
    io_tensors = [network.get_input(i) for i in range(network.num_inputs)] + \
                 [network.get_output(i) for i in range(network.num_outputs)]
    for t in io_tensors:
        print(f"  {t.name:<10} {t.dtype}  {tuple(t.shape)}")
        if t.dtype != trt.DataType.FLOAT:
            raise TypeError(
                f"Tensor '{t.name}' is {t.dtype}, expected FLOAT. "
                "Re-run AutoCast with the option that keeps I/O types."
            )

    # No precision flags: precision is defined by the graph (strong typing).
    config = builder.create_builder_config()

    print("Building engine (tactic selection may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mib = os.path.getsize(engine_path) / 2**20
    print(f"Engine saved: {engine_path} ({size_mib:.2f} MiB)")


if __name__ == "__main__":
    main()