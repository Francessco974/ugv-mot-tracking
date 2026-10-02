# scripts/t3_ByteTracker.py
import os
import argparse
import configparser
import numpy as np
import pandas as pd
from mot_tracking.tracker import run_tracker
from mot_tracking.boxes import tlwh_to_xyxy

def fmt(x):
    """0.35 -> '035' for folder names."""
    return str(x).replace(".", "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BYTE tracker on a MOT17 sequence with CLI configuration."
    )

    # Core tracker parameters
    parser.add_argument(
        "--use_kalman",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Kalman Filter motion model. Use --no-use_kalman to disable.",
    )
    parser.add_argument(
        "--match_algo",
        type=str,
        default="hungarian",
        choices=["hungarian", "greedy"],
        help="Data association algorithm (default: 'hungarian').",
    )
    parser.add_argument(
        "--output_box_type",
        type=str,
        default="posterior",
        choices=["posterior", "raw_det", "prediction"],
        help="Bounding box type to save in the output file (default: 'posterior').",
    )
    parser.add_argument(
        "--fuse_score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stages 1 and 3 use IoU*score as similarity (ByteTrack default). "
             "--no-fuse_score = plain IoU (ByteTrack --mot20 behaviour).",
    )

    # Association gates (one per stage)
    parser.add_argument("--iou_gate_high", type=float, default=0.1,
                        help="Stage 1 (confirmed+lost vs high): min similarity (default 0.1).")
    parser.add_argument("--iou_gate_low", type=float, default=0.5,
                        help="Stage 2 (confirmed vs low): min IoU (default 0.5).")
    parser.add_argument("--iou_gate_unconfirmed", type=float, default=0.3,
                        help="Stage 3 (tentative vs leftover high): min similarity (default 0.3).")

    # Detection score thresholds
    parser.add_argument("--high_confidence_threshold", type=float, default=0.60,
                        help="High detection confidence threshold (default: 0.6).")
    parser.add_argument("--low_confidence_threshold", type=float, default=0.10,
                        help="Low detection confidence threshold (default: 0.10).")
    parser.add_argument("--birth_threshold", type=float, default=None,
                        help="Min score to start a new track (default: high + 0.1).")

    # Paths
    parser.add_argument(
        "--seq_path",
        type=str,
        default=os.path.expanduser("~/ugv_data/mot/MOT17/train/MOT17-09-FRCNN"),
        help="Root path to the MOT sequence.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=os.path.expanduser(
            "~/ugv_data/mot/TrackEval/data/trackers/mot_challenge/MOT17-train"
        ),
        help="Base output directory for TrackEval.",
    )

    # Track lifecycle parameters
    parser.add_argument("--min_hints", type=int, default=2,
                        help="Hits required to confirm a track (default: 2).")
    parser.add_argument("--max_memory_s", type=float, default=1.0,
                        help="Time a lost track is kept before deletion, in seconds (default: 1.0).")

    args = parser.parse_args()
    if args.birth_threshold is None:
        args.birth_threshold = round(args.high_confidence_threshold + 0.1, 3)
    return args


def main():
    args = parse_args()

    # Sequence file paths
    test_path = os.path.join(args.seq_path, "det", "det.txt")
    seqinfo_path = os.path.join(args.seq_path, "seqinfo.ini")
    seq_name = os.path.basename(os.path.normpath(args.seq_path))

    # Dynamic naming: every parameter that drives the run is in the name
    tracker_mode = "kf" if args.use_kalman else "nokf"
    fuse_mode = "fs" if args.fuse_score else "nofs"
    tracker_name = (
        f"s3_{args.match_algo}_{tracker_mode}_{args.output_box_type}_{fuse_mode}"
        f"_gh{fmt(args.iou_gate_high)}_gl{fmt(args.iou_gate_low)}_gu{fmt(args.iou_gate_unconfirmed)}"
        f"_th{fmt(args.high_confidence_threshold)}_tl{fmt(args.low_confidence_threshold)}"
        f"_b{fmt(args.birth_threshold)}_h{args.min_hints}_ms{fmt(args.max_memory_s)}s"
    )

    tracker_output_path = os.path.join(
        args.output_base_dir, tracker_name, "data", f"{seq_name}.txt"
    )

    print("=" * 60)
    print(f"Running Tracker: {tracker_name}")
    print(f"Sequence: {seq_name}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"Output File: {tracker_output_path}")
    print("=" * 60)

    # Load detections
    col_df = [
        "frame_number",
        "id_number",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "confidence_score",
    ]
    detections = pd.read_csv(test_path, delimiter=",", header=None,
                             names=col_df, usecols=range(len(col_df)))

    # Read sequence length from seqinfo.ini
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)
    n_frames = int(cfg["Sequence"]["seqLength"])
    fps = float(cfg["Sequence"]["frameRate"])

    max_memory_frames = int(round(args.max_memory_s * fps))

    # Split detections by confidence (same band definition as T3.1: [low, high))
    bbox_cols = ["bb_left", "bb_top", "bb_width", "bb_height"]

    dets_high = detections[detections["confidence_score"] >= args.high_confidence_threshold]
    dets_low = detections[
        detections["confidence_score"].between(
            args.low_confidence_threshold,
            args.high_confidence_threshold,
            inclusive="left"
        )
    ]

    dets_high_by_frame = {
        int(f): np.array([tlwh_to_xyxy(r) for r in g[bbox_cols].to_numpy()])
        for f, g in dets_high.groupby("frame_number")
    }
    high_confidences_by_frame = {
        int(f): g["confidence_score"].to_numpy()
        for f, g in dets_high.groupby("frame_number")
    }
    dets_low_by_frame = {
        int(f): np.array([tlwh_to_xyxy(r) for r in g[bbox_cols].to_numpy()])
        for f, g in dets_low.groupby("frame_number")
    }

    rows = run_tracker(
        dets_high_by_frame=dets_high_by_frame,
        dets_low_by_frame=dets_low_by_frame,
        high_confidences_by_frame=high_confidences_by_frame,
        n_frames=n_frames,
        iou_gate_high=args.iou_gate_high,
        iou_gate_low=args.iou_gate_low,
        iou_gate_unconfirmed=args.iou_gate_unconfirmed,
        birth_threshold=args.birth_threshold,
        fuse_score=args.fuse_score,
        matcher=args.match_algo,
        use_kalman=args.use_kalman,
        output_box_type=args.output_box_type,
        min_hints=args.min_hints,
        max_memory=max_memory_frames,
    )

    # Save results in MOTChallenge format
    os.makedirs(os.path.dirname(tracker_output_path), exist_ok=True)
    np.savetxt(
        tracker_output_path,
        np.array(rows).reshape(-1, 10),
        delimiter=",",
        fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%d", "%d"],
    )

    print(f" [OK] Results successfully saved to: {tracker_output_path}\n")


if __name__ == "__main__":
    main()