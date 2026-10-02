# scripts/t2_SORT.py
import os
import argparse
import configparser
import numpy as np
import pandas as pd
from mot_tracking.tracker import run_tracker
from mot_tracking.boxes import tlwh_to_xyxy



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MOT tracker benchmark (SORT / IoU Tracker) with CLI configuration."
    )

    # Core Tracker Parameters (Default set to standard SORT)
    parser.add_argument(
        "--use_kalman",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Kalman Filter for motion model (default: True for standard SORT). Use --no-use_kalman to disable.",
    )
    parser.add_argument(
        "--match_algo",
        type=str,
        default="hungarian",
        choices=["hungarian", "greedy"],
        help="Data Association algorithm (default: 'hungarian').",
    )
    parser.add_argument(
        "--output_box_type",
        type=str,
        default="posterior",
        choices=["posterior", "raw_det", "prediction"],
        help="Bounding box type to save in the output file (default: 'posterior').",
    )

    # Thresholds
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="Minimum IoU threshold for matching (default: 0.3).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.05,
        help="Minimum detection confidence threshold (default: 0.05).",
    )

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
    
    # Track Lifecycle Parameters
    parser.add_argument(
        "--min_hints",
        type=int,
        default=2,
        help="Minimum consecutive hits required to confirm a track (default: 2).",
    )
    parser.add_argument(
        "--max_memory_s",
        type=float,
        default=1.0,
        help="Maximum frames to keep a lost track in memory before deletion (expressed in seconds) (default: 1.0 sec).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Sequence file paths
    test_path = os.path.join(args.seq_path, "det", "det.txt")
    seqinfo_path = os.path.join(args.seq_path, "seqinfo.ini")
    seq_name = os.path.basename(os.path.normpath(args.seq_path))

    # Dynamic naming to avoid accidental overwrites and stale files
    tracker_mode = "kf" if args.use_kalman else "nokf"
    iou_str = str(args.iou_threshold).replace(".", "")
    
    mem_s_str = str(args.max_memory_s).replace(".", "")
    
    tracker_name = (
        f"s2_{args.match_algo}_{tracker_mode}_{args.output_box_type}"
        f"_g{iou_str}_h{args.min_hints}_ms{mem_s_str}s"
    )

    tracker_output_path = os.path.join(
        args.output_base_dir, tracker_name, "data", f"{seq_name}.txt"
    )

    print("=" * 60)
    print(f"Running Tracker: {tracker_name}")
    print(f"Sequence: {seq_name}")
    print(f"Use Kalman: {args.use_kalman}")
    print(f"Matching Algorithm: {args.match_algo}")
    print(f"Output Box Type: {args.output_box_type}")
    print(f"Min Hints: {args.min_hints}")
    print(f"Max Memory in seconds: {args.max_memory_s}")
    print(f"IoU Threshold: {args.iou_threshold}")
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
    detections = pd.read_csv(test_path, delimiter=",", header=None, names=col_df)

    # Read sequence length from seqinfo.ini
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)
    n_frames = int(cfg["Sequence"]["seqLength"])
    fps = float(cfg["Sequence"]["frameRate"])
    
    max_memory_frames = int(round(args.max_memory_s * fps))

    # Filter detections by confidence threshold
    bbox_cols = ["bb_left", "bb_top", "bb_width", "bb_height"]
    dets = detections[detections["confidence_score"] > args.confidence_threshold]

    det_by_frame = {
        int(f): np.array([tlwh_to_xyxy(r) for r in g[bbox_cols].to_numpy()])
        for f, g in dets.groupby("frame_number")
    }

    # Run tracker with CLI arguments
    rows = run_tracker(
        det_by_frame=det_by_frame,
        n_frames=n_frames,
        iou_thr=args.iou_threshold,
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
        np.array(rows),
        delimiter=",",
        fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%d", "%d"],
    )

    print(f" [OK] Results successfully saved to: {tracker_output_path}\n")


if __name__ == "__main__":
    main()