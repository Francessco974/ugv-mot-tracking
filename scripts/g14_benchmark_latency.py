# mot_tracking/scripts/g14_benchmark_latency.py
"""G1.4 — latency benchmark of ContinuousDetector, per stage, with measurement conditions.

Sources:
  --image PATH      fixed image (pipeline only; no capture, no decode)
  --camera DEVICE   live V4L2 camera (adds capture time and timestamp analysis)

Per frame:
  capture   cap.read() (camera mode only: wait for frame + MJPG decode)
  stages    detector.last_timing_ns: preprocess / inference / postprocess
  detect    measured outside detect() -> 'overhead' = detect - sum(stages)
  loop      capture + detect

Every number is printed with its conditions (rule 4): platform, power mode,
jetson_clocks (declared), TensorRT version, git hash.
"""
import os
import sys
import json
import time
import argparse
import platform
import subprocess

import cv2
import numpy as np

from mot_tracking.detect.detect import ContinuousDetector

STAGES = ("preprocess", "inference", "postprocess")


def parse_args():
    p = argparse.ArgumentParser(description="G1.4 ContinuousDetector latency benchmark")
    p.add_argument("--engine", default=os.path.expanduser(
        "~/ugv_data/mot/models/yolov8n_mixed_pc.engine"))
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", default=os.path.expanduser("~/ugv_data/mot/fov_mjpg_720p.jpg"),
                     help="fixed input image")
    src.add_argument("--camera", default=None, help="V4L2 device, e.g. /dev/video0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fourcc", default="MJPG", help="MJPG or YUYV")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--conf", type=float, default=0.05, help="production value 0.05")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--jetson-clocks", choices=["on", "off", "unknown"], default="unknown",
                   help="declare jetson_clocks state (checking it needs sudo)")
    p.add_argument("--out", default=None, help="optional JSON summary path")
    p.add_argument("--resize", default=None, help="WxH applied to the fixed image, e.g. 1280x720")

    return p.parse_args()


# ---------------------------------------------------------------- conditions

def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def measurement_conditions(args):
    is_jetson = os.path.exists("/etc/nv_tegra_release")
    try:
        import tensorrt as trt
        trt_version = trt.__version__
    except Exception:
        trt_version = "n/a"

    power_mode = "n/a (not a Jetson)"
    if is_jetson:
        out = sh(["nvpmodel", "-q"])
        lines = [l for l in out.splitlines() if "Power Mode" in l]
        power_mode = lines[0].split(":", 1)[1].strip() if lines else "unknown"

    repo = os.path.dirname(os.path.abspath(__file__))
    git_hash = sh(["git", "-C", repo, "rev-parse", "--short", "HEAD"]) or "n/a"
    dirty = bool(sh(["git", "-C", repo, "status", "--porcelain"]))

    return {
        "platform": "jetson" if is_jetson else f"pc ({platform.machine()})",
        "kernel": platform.release(),
        "power_mode": power_mode,
        "jetson_clocks": args.jetson_clocks if is_jetson else "n/a",
        "tensorrt": trt_version,
        "opencv": cv2.__version__,
        "git": git_hash + (" (dirty)" if dirty else ""),
        "engine": os.path.basename(args.engine),
    }


# ---------------------------------------------------------------- statistics

def stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "std": float(a.std()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def row(name, s):
    if s is None:
        return f"{name:<14} | {'n/a':>8}"
    return (f"{name:<14} | {s['mean']:8.3f} | {s['p50']:8.3f} | {s['p95']:8.3f} | "
            f"{s['p99']:8.3f} | {s['max']:8.3f}")


def interval_analysis(times_ms, period_ms):
    """Successive differences; a gap > 1.5 periods counts as at least one dropped frame."""
    if len(times_ms) < 2:
        return None
    d = np.diff(np.asarray(times_ms, dtype=float))
    gaps = d > 1.5 * period_ms
    dropped = int(np.sum(np.round(d[gaps] / period_ms) - 1))
    return {"stats": stats(d), "gaps": int(gaps.sum()), "dropped_est": dropped}


# ---------------------------------------------------------------- camera

def open_camera(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))   # before size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    actual = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace"),
    }
    return cap, actual


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    if not os.path.exists(args.engine):
        raise FileNotFoundError(f"Engine missing: {args.engine}")

    cond = measurement_conditions(args)
    cap, cam_actual, fixed_frame = None, None, None
    if args.camera:
        cap, cam_actual = open_camera(args)
        source = (f"camera {args.camera} {cam_actual['fourcc']} "
                  f"{cam_actual['width']}x{cam_actual['height']} @ {cam_actual['fps']:.1f} fps")
    else:
        fixed_frame = cv2.imread(args.image)
        if fixed_frame is None:
            raise FileNotFoundError(f"Cannot read image {args.image}")
        if args.resize:
            rw, rh = map(int, args.resize.lower().split("x"))
            fixed_frame = cv2.resize(fixed_frame, (rw, rh), interpolation=cv2.INTER_AREA)
        h, w = fixed_frame.shape[:2]
        source = f"fixed image {os.path.basename(args.image)} {w}x{h}"

    print("=" * 78)
    print(" G1.4 ContinuousDetector latency benchmark")
    print("=" * 78)
    for k, v in cond.items():
        print(f"  {k:<14}: {v}")
    print(f"  {'source':<14}: {source}")
    print(f"  {'thresholds':<14}: conf {args.conf}, NMS IoU {args.iou}")
    print(f"  {'frames':<14}: {args.warmup} warmup + {args.iterations} measured")
    if cond["platform"] == "jetson" and "MAXN" not in cond["power_mode"].upper():
        print("  WARNING: power mode is not MAXN_SUPER — numbers are not G1.4 numbers")
    if cond["platform"] != "jetson":
        print("  NOTE: not a Jetson — use these numbers to test the script, not for G1.4")
    print("-" * 78)

    rec = {"capture": [], "detect": [], "loop": [], "overhead": []}
    rec.update({s: [] for s in STAGES})
    n_dets, host_capture_ms, driver_ts_ms = [], [], []

    try:
        with ContinuousDetector(engine_path=args.engine, conf_threshold=args.conf,
                                iou_threshold=args.iou) as detector:
            t_wall0 = None
            for n in range(args.warmup + args.iterations):
                t0 = time.perf_counter_ns()
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera read failed")
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                else:
                    frame, ts = fixed_frame, 0.0
                t1 = time.perf_counter_ns()
                dets = detector.detect(frame)
                t2 = time.perf_counter_ns()

                if n < args.warmup:
                    continue
                if t_wall0 is None:
                    t_wall0 = t0

                timing = detector.last_timing_ns
                stage_ns = [timing.get(s, 0) for s in STAGES]
                for s, v in zip(STAGES, stage_ns):
                    rec[s].append(v / 1e6)
                rec["detect"].append((t2 - t1) / 1e6)
                rec["overhead"].append((t2 - t1 - sum(stage_ns)) / 1e6)
                rec["loop"].append((t2 - t0) / 1e6)
                if cap is not None:
                    rec["capture"].append((t1 - t0) / 1e6)
                    host_capture_ms.append(t1 / 1e6)
                    driver_ts_ms.append(ts)
                n_dets.append(len(dets))
            t_wall1 = time.perf_counter_ns()
    finally:
        if cap is not None:
            cap.release()

    # ---- Results ----
    summary = {k: stats(v) for k, v in rec.items()}
    print(f"\n{'stage [ms]':<14} | {'mean':>8} | {'p50':>8} | {'p95':>8} | {'p99':>8} | {'max':>8}")
    print("-" * 78)
    if cap is not None or args.camera:
        print(row("capture", summary["capture"]))
    for s in STAGES:
        print(row(s, summary[s]))
    print(row("overhead", summary["overhead"]))
    print("-" * 78)
    print(row("detect", summary["detect"]))
    print(row("loop", summary["loop"]))
    print("=" * 78)

    wall_s = (t_wall1 - t_wall0) / 1e9
    det = stats(n_dets)
    print(f"Detect only : {1000 / summary['detect']['mean']:.1f} fps from mean, "
          f"{1000 / summary['detect']['p99']:.1f} fps from p99")
    print(f"End to end  : {args.iterations / wall_s:.1f} fps wall clock "
          f"({args.iterations} frames in {wall_s:.2f} s)")
    print(f"Detections  : mean {det['mean']:.1f}, max {det['max']:.0f} per frame "
          f"(conf {args.conf})")

    timestamps = None
    if args.camera:
        period = 1000.0 / cam_actual["fps"] if cam_actual["fps"] > 0 else 1000.0 / args.fps
        host = interval_analysis(host_capture_ms, period)
        drv = interval_analysis(driver_ts_ms, period) if any(driver_ts_ms) else None
        print(f"\nFrame intervals (expected {period:.2f} ms):")
        for name, a in (("host clock", host), ("driver ts", drv)):
            if a is None:
                print(f"  {name:<11}: not provided by this backend")
                continue
            s = a["stats"]
            print(f"  {name:<11}: mean {s['mean']:.2f}  p50 {s['p50']:.2f}  p99 {s['p99']:.2f}  "
                  f"max {s['max']:.2f} ms | gaps {a['gaps']} | dropped ~{a['dropped_est']}")
        timestamps = {"expected_period_ms": period, "host": host, "driver": drv}
        if summary["loop"]["mean"] > period:
            print("  NOTE: loop slower than camera period — frames queue or drop; "
                  "latency grows (see L6 one-slot buffer)")

    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump({"conditions": cond, "source": source, "conf": args.conf, "iou": args.iou,
                       "iterations": args.iterations, "stages_ms": summary,
                       "detections": det, "fps_wall": args.iterations / wall_s,
                       "camera": cam_actual, "timestamps": timestamps}, f, indent=2)
        print(f"\nSummary saved: {args.out}")


if __name__ == "__main__":
    main()