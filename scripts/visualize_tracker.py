import os
import glob
import argparse
import cv2
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Renderizza un video delle tracce MOT.")
    parser.add_argument("--seq_path", type=str, required=True)
    parser.add_argument("--res_file", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="output_tracking.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_frame", type=int, default=180)
    parser.add_argument("--end_frame", type=int, default=330)
    return parser.parse_args()


def get_color(idx):
    np.random.seed(int(idx))
    return tuple(map(int, np.random.randint(0, 255, size=3)))


def open_writer(path, fps, size):
    """Try H.264 first; fall back to mp4v with a warning if the encoder isn't available."""
    for tag in ("avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            print(f"[OK] Using codec '{tag}' (H.264)")
            return writer
    print("[WARN] H.264 encoder not available in this OpenCV build — falling back to mp4v.")
    print("       Re-encode afterward: ffmpeg -i <out> -c:v libx264 -pix_fmt yuv420p -an <final>.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)


def main():
    args = parse_args()
    OUT_W, OUT_H = 1280, 720

    cols = ["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"]
    if not os.path.exists(args.res_file):
        raise FileNotFoundError(f"File risultati non trovato: {args.res_file}")
    df = pd.read_csv(args.res_file, header=None, names=cols)
    tracks_by_frame = {frame: g for frame, g in df.groupby("frame")}

    img_dir = os.path.join(args.seq_path, "img1")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"Nessuna immagine .jpg trovata in {img_dir}")

    # original resolution BEFORE slicing/resizing — this is what the tracker's
    # box coordinates are expressed in
    first_img = cv2.imread(img_paths[0])
    orig_h, orig_w = first_img.shape[:2]
    scale_x = OUT_W / orig_w
    scale_y = OUT_H / orig_h
    print(f"Source: {orig_w}x{orig_h} -> {OUT_W}x{OUT_H} (scale_x={scale_x:.4f}, scale_y={scale_y:.4f})")

    img_paths = img_paths[args.start_frame - 1:args.end_frame]

    out = open_writer(args.output_video, args.fps, (OUT_W, OUT_H))

    print(f"Generazione video in corso per {len(img_paths)} frame...")
    for i, img_path in enumerate(img_paths):
        frame_idx = args.start_frame + i          # true frame number, matches tracker file
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (OUT_W, OUT_H))

        if frame_idx in tracks_by_frame:
            for _, row in tracks_by_frame[frame_idx].iterrows():
                track_id = int(row["id"])
                # scale box coords into the resized frame's pixel space
                x = int(row["x"] * scale_x)
                y = int(row["y"] * scale_y)
                w = int(row["w"] * scale_x)
                h = int(row["h"] * scale_y)

                color = get_color(track_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = f"ID: {track_id}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w, y), color, -1)
                cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out.write(frame)

    out.release()
    print(f" [OK] Video salvato con successo in: {args.output_video}")


if __name__ == "__main__":
    main()