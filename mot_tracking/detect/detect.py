# mot_tracking/mot_tracking/detect/detect.py
import os
import numpy as np
from mot_tracking.detect.letterbox import pre_process_image
from mot_tracking.detect.trt_infer import TRTInfer
from mot_tracking.detect.postprocess import decode, NMS, unletterbox_boxes, clip_P1P2

ENGINE_PATH = os.environ.get(
    "MOT_ENGINE", os.path.expanduser("~/ugv_data/mot/models/yolov8n_mixed_pc.engine"))


class ContinuousDetector:
    def __init__(self, engine_path: str = ENGINE_PATH,
                 conf_threshold: float = 0.05, iou_threshold: float = 0.7):
        self.engine = TRTInfer(engine_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 frame -> (K, 5) [x1, y1, x2, y2, score] in frame pixels."""
        h, w = frame.shape[:2]
        tensor, (scale, pad_w_l, pad_h_u) = pre_process_image(frame)

        out = self.engine.infer(tensor)
        boxes, scores = decode(out, self.conf_threshold)
        dets = NMS(boxes, scores, self.iou_threshold)
        dets = unletterbox_boxes(dets, scale, pad_w_l, pad_h_u)
        return clip_P1P2(dets, (h, w))

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()