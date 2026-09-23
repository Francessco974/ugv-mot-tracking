# mot_tracking/tracker.py
import numpy as np
from mot_tracking.iou import compute_IoU, to_BB_coordinates


def greedy_match(iou, thr):
    """Global greedy: repeatedly commit the highest remaining IoU pair >= thr."""
    iou = iou.copy()
    matches = []
    while iou.size:
        # np.argmax give you a 1D index, unravel_index give you the 2D coords
        t, d = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[t, d] < thr:          # best remaining is below gate -> nothing else qualifies
            break
        matches.append((t, d))
        iou[t, :] = -1.0             # track row consumed
        iou[:, d] = -1.0             # detection column consumed
    return matches


def run_greedy_tracker(det_by_frame, n_frames, iou_thr):
    """det_by_frame: {frame: (M,4) array in x1,y1,x2,y2}. Returns MOT-format rows."""
    track_ids = np.empty(0, dtype=int)     # persistent identity of each row
    track_boxes = np.empty((0, 4))         # last seen box of each row
    next_id = 1
    rows = []

    for frame in range(1, n_frames + 1):
        dets = det_by_frame.get(frame, np.empty((0, 4)))

        matches = []
        if len(track_boxes) and len(dets):
            matches = greedy_match(compute_IoU(track_boxes, dets), iou_thr)

        # continuing tracks keep their PERSISTENT id, not their row index
        new_ids = [track_ids[t] for t, _ in matches]
        new_boxes = [dets[d] for _, d in matches]

        # unmatched detections are born; unmatched tracks simply aren't carried over (dead)
        matched_dets = {d for _, d in matches}
        for d in range(len(dets)):
            if d not in matched_dets:
                new_ids.append(next_id)
                new_boxes.append(dets[d])
                next_id += 1

        track_ids = np.array(new_ids, dtype=int)
        track_boxes = np.array(new_boxes, dtype=float).reshape(-1, 4)

        for tid, box in zip(track_ids, track_boxes):
            rows.append([frame, tid, *to_BB_coordinates(box), 1, -1, -1, -1])

    return rows