# mot_tracking/mot_tracking/detect/detect.py
import os
import time
import numpy as np
from typing import Dict
from mot_tracking.detect.letterbox import pre_process_image
from mot_tracking.detect.trt_infer import TRTInfer
from mot_tracking.detect.postprocess import decode, NMS, unletterbox_boxes, clip_P1P2

ENGINE_PATH = os.environ.get(
    "MOT_ENGINE", os.path.expanduser("~/ugv_data/mot/models/yolov8n_mixed_pc.engine"))


class ContinuousDetector:
    def __init__(self, engine_path: str = ENGINE_PATH,
                 conf_threshold: float = 0.05, iou_threshold: float = 0.7, max_det=300):
        self.engine = TRTInfer(engine_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        # Timing dictionary in nanoseconds
        self.last_timing_ns: Dict[str, int] = {
            "preprocess": 0,
            "inference": 0,
            "postprocess": 0,
        }

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 frame -> (K, 5) [x1, y1, x2, y2, score] in frame pixels."""
        h, w = frame.shape[:2]
        
        # 1. Preprocess
        t0 = time.perf_counter_ns()
        tensor, (scale, pad_w_l, pad_h_u) = pre_process_image(frame)
        t1 = time.perf_counter_ns()
        
        # 2. Inference
        out = self.engine.infer(tensor)
        t2 = time.perf_counter_ns()
        
        boxes, scores = decode(out, self.conf_threshold)
        dets = NMS(boxes, scores, self.iou_threshold, max_det=self.max_det)
        dets = unletterbox_boxes(dets, scale, pad_w_l, pad_h_u)
        clipped_dets = clip_P1P2(dets, (h, w))
        t3 = time.perf_counter_ns()
        
        # Store stage timings in ns
        self.last_timing_ns["preprocess"] = t1 - t0
        self.last_timing_ns["inference"] = t2 - t1
        self.last_timing_ns["postprocess"] = t3 - t2
        
        return clipped_dets

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()