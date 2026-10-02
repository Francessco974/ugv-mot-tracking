# mot_tracking/scripts/t5_pc_detection.py 
import os
import cv2
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.open_in_browser"] = False
matplotlib.rcParams["webagg.port"] = 8988
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mot_tracking.detect.detect import ContinuousDetector
from mot_tracking.detect.letterbox import letter_box

paths = [os.path.expanduser("~/ugv_data/mot/fov_mjpg_720p.jpg"),
         os.path.expanduser("~/ugv_data/mot/fov_yuyv_480p.png")]


def draw(ax, img_rgb, dets, title):
    ax.imshow(img_rgb)
    ax.set_title(title)
    for x1, y1, x2, y2, s in dets[:, :5]:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, ec="lime"))
        ax.text(x1, y1 - 4, f"{s:.2f}", color="lime", fontsize=9)


with ContinuousDetector() as det:
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"Not found: {p}")
            continue

        dets = det.detect(frame)
        lb, scale, (pw, ph) = letter_box(frame)
        print(f"{os.path.basename(p)}: shape={frame.shape} scale={scale:.4f} pad=({pw},{ph})")
        print(dets.round(1))

        # Same detections mapped forward into letterbox space, for the debug view
        dets_lb = dets.copy()
        dets_lb[:, [0, 2]] = dets_lb[:, [0, 2]] * scale + pw
        dets_lb[:, [1, 3]] = dets_lb[:, [1, 3]] * scale + ph

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
        draw(a1, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dets, f"{os.path.basename(p)} (original)")
        draw(a2, cv2.cvtColor(lb, cv2.COLOR_BGR2RGB), dets_lb, "letterbox 640x640")
        print("http://localhost:8988/  (Ctrl+C for next image)")
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        plt.close(fig)