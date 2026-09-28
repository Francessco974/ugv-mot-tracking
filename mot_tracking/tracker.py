# mot_tracking/tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment

from mot_tracking.iou import compute_IoU, to_BB_coordinates, to_IoU_coordinates
from mot_tracking.kalman import to_kf_coordinates, from_kf_to_bb, SortKF, to_kf_from_IoU, to_IoU_from_kf


# ── Matchers ───────────────────────────────────────────────────────────────
# Both take a (T, D) SIMILARITY matrix (IoU, or IoU*score) and a SCALAR gate.

def greedy_match(sim, thr):
    """Greedy: repeatedly take the best remaining pair while it passes the gate.
    With a scalar gate, stopping at the first max below thr is correct:
    every remaining value is <= that max."""
    if sim.size == 0:
        return []

    work = sim.astype(float).copy()
    matches = []
    while True:
        t, d = np.unravel_index(np.argmax(work), work.shape)
        if work[t, d] < thr:              # also catches -inf (all used)
            break
        matches.append((int(t), int(d)))
        work[t, :] = -np.inf
        work[:, d] = -np.inf
    return matches


def hungarian_match(sim, thr):
    """Optimal assignment with the gate inside the optimisation.
    Pairs below thr are set to 0 before solving: a 0 pair adds nothing to the
    total, exactly like leaving both sides unmatched, so the solver maximises
    the total similarity over valide pairs only. The post-filter then drops any
    masked pair the solver still returned."""
    if sim.size == 0:
        return []

    valid = sim >= thr
    gated = np.where(valid, sim, 0.0)
    rows, cols = linear_sum_assignment(gated, maximize=True)
    # fancy indexing → output is array([True, True, False, ...])
    # can be use to filter optimizer output
    keep = valid[rows, cols]
    
    return list(zip(rows[keep].tolist(), cols[keep].tolist()))


def associate(predicted_boxes, track_ids, dets, confs, gate, matcher_function, fuse_score):
    """One association stage.

    predicted_boxes : {id: box} Kalman predictions of this frame (IoU coords)
    track_ids       : IDs allowed to compete in this stage
    dets, confs     : (D, 4) boxes and (D,) scores (confs unused if not fused)
    fuse_score      : True -> similarity = IoU * score (ByteTrack fuse_score)

    Returns ({id: det_idx}, unmatched_ids, unmatched_det_idxs).
    """
    if len(track_ids) == 0 or len(dets) == 0:
        return {}, list(track_ids), list(range(len(dets)))

    boxes = np.array([predicted_boxes[i] for i in track_ids])
    sim = compute_IoU(boxes, dets)
    if fuse_score:
        sim = sim * np.asarray(confs).reshape(1, -1)

    pairs = matcher_function(sim, gate)
    matched = {track_ids[t]: d for t, d in pairs}
    used = set(matched.values())
    unmatched_ids = [i for i in track_ids if i not in matched]
    unmatched_dets = [d for d in range(len(dets)) if d not in used]
    return matched, unmatched_ids, unmatched_dets


# ── Tracker loop ───────────────────────────────────────────────────────────

def run_tracker(
    dets_high_by_frame,
    dets_low_by_frame,
    high_confidences_by_frame,
    n_frames,
    iou_gate_high=0.1,          # stage 1 gate on IoU*s  (ByteTrack: 1 - match_thresh 0.9)
    iou_gate_low=0.5,           # stage 2 gate on IoU    (ByteTrack: 1 - 0.5)
    iou_gate_unconfirmed=0.3,   # stage 3 gate on IoU*s  (ByteTrack: 1 - 0.7)
    birth_threshold=0.7,        # min score to start a track (ByteTrack: track_thresh + 0.1)
    fuse_score=True,            # IoU*s in stages 1 and 3 (ByteTrack default, off with --mot20)
    matcher='hungarian',
    use_kalman=True,
    output_box_type='posterior',
    min_hints=2,
    max_memory=1,
    check_invariants=False,     # True: assert the per-frame invariants (slower; for verification runs)
    stats=None,                 # optional dict, filled with per-stage counters and stage-2 matches
    ):

    if matcher == 'hungarian':
        matcher_function = hungarian_match
    elif matcher == 'greedy':
        matcher_function = greedy_match
    else:
        raise ValueError("matcher is not valid. Possible options: ['hungarian' , 'greedy']")

    tracker_master = TrackerMaster(
        use_kalman=use_kalman,
        min_hints=min_hints,
        max_memory=max_memory
    )

    rows = []

    for frame in range(1, n_frames + 1):
        dets_high = dets_high_by_frame.get(frame, np.empty((0, 4)))
        conf_high = high_confidences_by_frame.get(frame, np.empty((0,))).reshape(-1)
        dets_low = dets_low_by_frame.get(frame, np.empty((0, 4)))

        # Predict all tracks once; pools are fixed before any update, so a
        # track re-found in stage 1 cannot re-enter a later stage.
        predicted_boxes = tracker_master.predict()
        confirmed_ids = tracker_master.ids_in_state('confirmed')
        lost_ids = tracker_master.ids_in_state('lost')
        tentative_ids = tracker_master.ids_in_state('tentative')

        # Stage 1 — confirmed + lost  vs  high detections
        m1, u_ids1, u_det1 = associate(
            predicted_boxes, confirmed_ids + lost_ids, dets_high, conf_high,
            iou_gate_high, matcher_function, fuse_score)
        tracker_master.update({i: dets_high[d] for i, d in m1.items()})

        # Stage 2 — confirmed (NOT lost) unmatched in stage 1  vs  low detections
        confirmed_set = set(confirmed_ids)
        
        # u_ids1 stands for unmatched id stage 1
        stage2_ids = [i for i in u_ids1 if i in confirmed_set]
        m2, _, _ = associate(
            predicted_boxes, stage2_ids, dets_low, None,
            iou_gate_low, matcher_function, fuse_score=False)
        tracker_master.update({i: dets_low[d] for i, d in m2.items()})

        # Stage 3 — tentative  vs  high detections left over from stage 1
        # u_det1 stands for: unmatched detection stage 1 (indeces)
        left_dets = dets_high[u_det1]
        left_conf = conf_high[u_det1]
        m3, _, u_det3 = associate(
            predicted_boxes, tentative_ids, left_dets, left_conf,
            iou_gate_unconfirmed, matcher_function, fuse_score)
        tracker_master.update({i: left_dets[d] for i, d in m3.items()})

        if check_invariants:
            _check_frame_invariants(
                frame, predicted_boxes, confirmed_ids, lost_ids, tentative_ids,
                (m1, dets_high, conf_high, iou_gate_high, fuse_score),
                (m2, dets_low, None, iou_gate_low, False),
                (m3, left_dets, left_conf, iou_gate_unconfirmed, fuse_score),
                u_det1)

        # Life cycle: unmatched tentative -> deleted, confirmed -> lost,
        # lost beyond max_memory -> deleted.
        tracker_master.update_life()
        tracker_master.life_check()

        # Births — only leftovers of stage 3, and only if confident enough
        births = [d for d in u_det3 if left_conf[d] >= birth_threshold]
        for d in births:
            tracker_master.new_track(left_dets[d])

        if stats is not None:
            for key, n in (("stage1", len(m1)), ("stage2", len(m2)),
                           ("stage3", len(m3)), ("births", len(births))):
                stats[key] = stats.get(key, 0) + n
            stats.setdefault("stage2_matches", []).extend(
                (frame, dets_low[d].copy()) for d in m2.values())

        rows.extend(tracker_master.mot_format_output(frame, output_box_type=output_box_type))

    return rows


def _check_frame_invariants(frame, predicted_boxes, confirmed_ids, lost_ids,
                            tentative_ids, s1, s2, s3, u_det1):
    """Assert what must be true after the three association stages of one frame.

    s1, s2, s3 = (matches {id: det_idx}, dets, confs, gate, fuse) of each stage.
    Raises AssertionError with the frame number on the first violation.
    """
    pools = [set(confirmed_ids), set(lost_ids), set(tentative_ids)]
    
    # Every existing track is in exactly one pool, and the pools cover all predicted tracks.
    assert sum(len(p) for p in pools) == len(set().union(*pools)), f"frame {frame}: pools overlap"
    assert set().union(*pools) == set(predicted_boxes), f"frame {frame}: pools != predicted tracks"

    # unpacking, they follow :  (m1, dets_high, conf_high, iou_gate_high, fuse_score),
    (m1, d1, c1, g1, f1), (m2, d2, c2, g2, f2), (m3, d3, c3, g3, f3) = s1, s2, s3
    
    # Each stage only matches tracks from its own pool.
    # set(m1) → set of the dictionary value
    # pools[0] | pools[1]: Is the union between set(confirmed_ids) and set(lost_ids)
    
    assert set(m1) <= pools[0] | pools[1], f"frame {frame}: stage 1 matched a tentative track"
    assert set(m2) <= pools[0] - set(m1), f"frame {frame}: stage 2 matched a lost/tentative/stage-1 track"
    assert set(m3) <= pools[2], f"frame {frame}: stage 3 matched a non-tentative track"
    
    # No track updated twice in one frame.
    assert not (set(m1) & set(m2) or set(m1) & set(m3) or set(m2) & set(m3)), \
        f"frame {frame}: a track was updated twice"
    
    # No detection used twice inside a stage; stage 3 only sees stage-1 leftovers.
    for k, m in (("1", m1), ("2", m2), ("3", m3)):
        assert len(set(m.values())) == len(m), f"frame {frame}: stage {k} reused a detection"
    assert len(d3) == len(u_det1), f"frame {frame}: stage 3 input is not the stage-1 leftover"
    # Every accepted pair respects its stage's gate (recomputed independently).
    for k, (m, dets, confs, gate, fuse) in (("1", s1), ("2", s2), ("3", s3)):
        for tid, d in m.items():
            sim = compute_IoU(np.asarray([predicted_boxes[tid]]), dets[d:d + 1])[0, 0]
            if fuse:
                sim *= confs[d]
            assert sim >= gate - 1e-9, f"frame {frame}: stage {k} pair below gate ({sim:.3f} < {gate})"


P0 = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])


class TrackerMaster:
    def __init__(self, use_kalman=True, min_hints=2, max_memory=1):
        self.tracked = {}
        self.next_id = 1
        self.use_kalman = use_kalman
        self.min_hints = min_hints
        self.max_memory = max_memory

    def new_track(self, boxes: np.ndarray):
        """
        Boxes is in IoU coordinates
        """
        self.tracked[self.next_id] = EstimatorLifeManager(self.next_id,
                                                          boxes,
                                                          use_kalman=self.use_kalman,
                                                          min_hints=self.min_hints,
                                                          max_memory=self.max_memory)
        self.next_id += 1

    def predict(self):
        return {key: value.predict() for key, value in self.tracked.items()}

    def ids_in_state(self, state):
        return [key for key, estimator in self.tracked.items() if estimator.state == state]

    def get_lost_keys(self):
        return self.ids_in_state('lost')

    def tracked_ids(self):
        return list(self.tracked.keys())

    def update(self, matches):
        """
        matches: {id : bb}
        """
        for id, bb_measure in matches.items():
            self.tracked[id].update(bb_measure)

    def update_life(self):
        for estimator in self.tracked.values():
            estimator.update_status()

    def life_check(self):
        """
        Remove old tracked
        """
        self.tracked = {k: v for k, v in self.tracked.items() if v.state != 'deleted'}

    def get_position_estimate(self, ids, output_box_type='posterior'):
        return self.tracked[ids].get_position_estimate(output_box_type)

    def mot_format_output(self, frame_number, output_box_type='posterior'):
        """"
        Return a series of list of rows
        """
        rows = []
        for track_id, estimator in self.tracked.items():

            if estimator.state == 'confirmed':
                rows.append([
                    frame_number,
                    track_id,
                    *estimator.get_position_estimate(output_box_type),
                    1, -1, -1, -1
                ])
        return rows


class EstimatorLifeManager:
    def __init__(self, ID, measurement, use_kalman=True, min_hints=2, max_memory=1):
        """
        measurement is in IoU coordinates
        """
        self.id = ID
        self.state = 'tentative'
        self.tracking_count = 1
        self.age = 0
        self.use_kalman = use_kalman
        self.min_hints = min_hints
        self.max_memory = max_memory

        # allow no KF
        self.last_measurement = measurement.copy()

        # Keep track of valid prediction if we want to use it instead of posterior
        self.current_prediction = measurement.copy()

        if self.tracking_count >= self.min_hints:
            self.state = 'confirmed'

        if self.use_kalman:
            self.localisation_estimator = SortKF(dt=1) # frame to frame, not frame to time
            self.localisation_estimator.initiate(to_kf_from_IoU(measurement), P0)

    def update(self, measurement: np.ndarray):
        """
        Measurement is in IoU coordinates.
        Called only if a there is a match at frame t
        """
        self.tracking_count += 1
        self.age = 0
        self.last_measurement = measurement.copy()

        if self.use_kalman:
            self.localisation_estimator.update(to_kf_from_IoU(measurement))

        if self.state == 'lost':                       # re-found after coasting
            self.state = 'confirmed'
        elif self.state == 'tentative' and self.tracking_count >= self.min_hints:
            self.state = 'confirmed'

    def predict(self):
        """
        Return np.array(4 elements in IoU coordinates).
        Salve last prediction
        """
        self.age += 1
        if self.use_kalman:
            pred_kf = self.localisation_estimator.predict().copy()
            self.current_prediction = to_IoU_from_kf(pred_kf)
        else:
            # No Kalman: dumb floor
            self.current_prediction = self.last_measurement.copy()
        return self.current_prediction

    def update_status(self):
        if self.age == 0:                              # matched this frame: nothing to do
            return
        if self.state == 'tentative':                  # unconfirmed tracks die on first miss
            self.state = 'deleted'
        elif self.age >= self.max_memory:
            self.state = 'deleted'
        else:
            self.state = 'lost'

    def get_position_estimate(self, output_box_type='posterior'):
        """Return bounding box in MOT coordinates"""

        if self.age > 0:
            return to_BB_coordinates(self.current_prediction)

        # If matched age = 0
        if output_box_type == 'raw_det' or not self.use_kalman:
            return to_BB_coordinates(self.last_measurement)
        elif output_box_type == 'prediction':
            return to_BB_coordinates(self.current_prediction)
        elif output_box_type == 'posterior':
            return from_kf_to_bb(self.localisation_estimator.state_[:4].tolist())
        else:
            raise ValueError(f"output_box_type non valido: {output_box_type}")