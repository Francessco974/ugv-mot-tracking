# scripts/t1_explore.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mot_tracking.utils import gt_visual_rectangle, dt_visual_rectangle
from mot_tracking.iou import to_IoU_coordinates, to_BB_coordinates,compute_IoU


TEST_PATH = os.path.expanduser("~/ugv_data/mot/MOT17/train/MOT17-09-FRCNN/det/det.txt")
GT_PATH = os.path.expanduser("~/ugv_data/mot/MOT17/train/MOT17-09-FRCNN/gt/gt.txt")
SEQ_PATH = os.path.expanduser("~/ugv_data/mot/MOT17/train/MOT17-09-FRCNN")

DUMB_TRACKER_FILE_PATH =  os.path.expanduser("~/ugv_data/mot/TrackEval/data/trackers/mot_challenge/MOT17-train/t12_gt_selftest/data/MOT17-09-FRCNN.txt")

GREEDY_TRACKER_FILE_PATH =  os.path.expanduser("~/ugv_data/mot/TrackEval/data/trackers/mot_challenge/MOT17-train/t13_greedy_tracker/data/MOT17-09-FRCNN.txt")

COL_DF = ['frame_number',
           'id_number',
           'bb_left',
           'bb_top',
           'bb_width',
           'bb_height',
           'confidence_score'
           ]

COL_GT = COL_DF + ['bb_class', 'Visibility']

   
detections = pd.read_csv(TEST_PATH, delimiter=",", header=None)
detections.columns = COL_DF

ground_truth = pd.read_csv(GT_PATH, delimiter=",", header=None)
ground_truth.columns = COL_GT

frame_counter = detections['frame_number'].value_counts().reset_index()
frame_counter.columns = ['frame_number', 'count']

if False:
    print(frame_counter.describe())
    
# Show some images
if False:
    for frame in [1, 100, 200]:
        img_path = f"{SEQ_PATH}/img1/{frame:06d}.jpg"

        image = plt.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(image)


        frame_gt = ground_truth.groupby('frame_number').get_group(frame)
        frame_dt = detections.groupby('frame_number').get_group(frame)

        for row in frame_gt.itertuples():
            rect = gt_visual_rectangle(row.bb_left,row.bb_top,row.bb_width,row.bb_height,row.bb_class)
            ax.add_patch(rect)

        for row in frame_dt.itertuples():
            rect = dt_visual_rectangle(row.bb_left,row.bb_top,row.bb_width,row.bb_height,600)
            ax.add_patch(rect)


        plt.show()
        plt.close()

perfect_tracker_res = ground_truth[
    (ground_truth.bb_class == 1) & (ground_truth.confidence_score == 1)
].copy()

# TrackEval expect:
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
perfect_tracker_res.drop(columns=['bb_class', 'Visibility'], inplace=True)
perfect_tracker_res[['confidence_score']] = 1.0
perfect_tracker_res[['x', 'y', 'z']] = -1

if False:
    # Ensure directory path exists
    os.makedirs(os.path.dirname(DUMB_TRACKER_FILE_PATH), exist_ok=True)
    perfect_tracker_res.to_csv(DUMB_TRACKER_FILE_PATH, sep=',', index=False, header=False)
    
# T1_3

from mot_tracking.tracker import run_greedy_tracker
import configparser
cfg = configparser.ConfigParser()

# Get sequence directly from its file
cfg.read(f"{SEQ_PATH}/seqinfo.ini")
n_frames = int(cfg["Sequence"]["seqLength"])


CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.3


bbox_cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
dets = detections[detections["confidence_score"] > CONFIDENCE_THRESHOLD]

det_by_frame = {
    int(f): np.array([to_IoU_coordinates(r) for r in g[bbox_cols].to_numpy()])
    for f, g in dets.groupby("frame_number")
}

rows = run_greedy_tracker(det_by_frame, n_frames, IOU_THRESHOLD)

os.makedirs(os.path.dirname(GREEDY_TRACKER_FILE_PATH), exist_ok=True)
np.savetxt(GREEDY_TRACKER_FILE_PATH, np.array(rows), delimiter=",",
           fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%d", "%d"])
