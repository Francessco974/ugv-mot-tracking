# mot_tracking/tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment

from mot_tracking.iou import compute_IoU, to_BB_coordinates, to_IoU_coordinates
from mot_tracking.kalman import to_kf_coordinates, from_kf_to_bb, SortKF, to_kf_from_IoU, to_IoU_from_kf

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


def hungarian_match(iou, thr):
    cost_matrix =  -iou + 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    good = cost_matrix[row_ind, col_ind] < (1 - thr)
    matches = list(zip(row_ind[good], col_ind[good]))
    return matches
    

# For t1 usage
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


def run_tracker(
    det_by_frame, 
    n_frames, 
    iou_thr=0.5, 
    matcher='hungarian', 
    use_kalman=True, 
    output_box_type='posterior',
    min_hints=2,
    max_memory=1
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
        # measurement
        dets = det_by_frame.get(frame, np.empty((0, 4)))
        
        matches = [] 
        predicted_boxes =  tracker_master.predict()
        predicted_idx = list(predicted_boxes.keys())
        # if → update
        if len(predicted_boxes) and len(dets):
            track_boxes = np.array(list(predicted_boxes.values()))
            matches = matcher_function(compute_IoU(track_boxes, dets), iou_thr)
            
            # matches give you index not ID 
            measurements = { predicted_idx[tracked_id] :  dets[detected_idx] for (tracked_id, detected_idx) in matches}
            tracker_master.update(measurements)

        tracker_master.update_life()
        tracker_master.life_check()

        # add new bb
        dected_matched_idxs = [d for _, d in matches]
        for detection_idx in range(len(dets)): 
            if detection_idx not in dected_matched_idxs:
                tracker_master.new_track(dets[detection_idx])
                
                
        rows.extend(tracker_master.mot_format_output(frame, output_box_type=output_box_type))
            

    return rows
    


P0 = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])


class TrackerMaster:
    def __init__(self, use_kalman=True, min_hints=2, max_memory=1):
        self.tracked = {}
        self.next_id = 1
        self.use_kalman = use_kalman
        self.min_hints = min_hints
        self.max_memory = max_memory
    
    def new_track (self, boxes : np.ndarray):    
        """ 
        Boxes is in IoU coordinates
        """
        self.tracked[self.next_id] = EstimatorLifeManager(self.next_id,
                                                          boxes, 
                                                          use_kalman=self.use_kalman,
                                                          min_hints = self.min_hints,
                                                          max_memory = self.max_memory)
        self.next_id += 1
        
        
    def predict(self):
        return {key : value.predict() for key, value in self.tracked.items()}
        
    
    def update (self, matches):
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
        
    
    def get_position_estimate(self, ids):
        return self.tracked[ids].get_position_estimate()
        
       
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
            self.localisation_estimator = SortKF(dt = 1)
            self.localisation_estimator.initiate(to_kf_from_IoU(measurement), P0)
    
    def update(self, measurement : np.ndarray ):
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
        
        self.age +=1
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