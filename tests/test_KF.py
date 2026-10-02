import numpy as np
import pytest
from mot_tracking.kalman import SortKF
from mot_tracking.boxes import tlwh_to_cxcysr, cxcysr_to_tlwh

def test_KF_filter_no_noise():

    velocity = 5.0
    dt = 1.0
    sim_time = 50.0
    samples = int(sim_time//dt)

    state = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    measurement = np.array([0, 0, 1, 1], dtype=float)

    KalmanFilter = SortKF(dt=dt)
    KalmanFilter.set_initial_state_(state)

    for k in range(samples):
        KalmanFilter.predict()
        measurement += np.array([velocity * dt, 0, 0, 0])
        KalmanFilter.update(measurement)
        
    assert(np.isclose(KalmanFilter.state_, 
            np.array([velocity * samples * dt, 0, 1, 1, velocity, 0, 0]), atol = 0.05).all())
    

def test_KF_filter_noise():
    velocity = 5.0
    dt = 1.0
    sim_time = 50.0
    samples = int(sim_time // dt)

    state = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    measurement = np.array([0, 0, 1, 1], dtype=float)

    KalmanFilter = SortKF(dt=dt)
    KalmanFilter.initiate(measurement, np.eye(7))

    # Set seed for deterministic pytest execution
    np.random.seed(42)

    for k in range(samples):
        KalmanFilter.predict()
        # Zero-mean Gaussian measurement noise
        noise = np.random.normal(0, 0.01, size=4)
        measurement += np.array([velocity * dt, 0, 0, 0]) + noise
        KalmanFilter.update(measurement)

    # Velocity (state_[4]) converges quickly and stays tight (~5.0)
    # Positions smooth out the noise over 50 steps
    expected_state = np.array([velocity * samples * dt, 0, 1, 1, velocity, 0, 0])
    
    assert np.isclose(KalmanFilter.state_, expected_state, atol=0.2).all()
    
    
    
def test_coordinates_transform():
    a = np.array([1,2,3,4], dtype=float)
    assert(np.isclose(tlwh_to_cxcysr(cxcysr_to_tlwh(a)), a, atol = 0.05).all())
    assert(np.isclose(cxcysr_to_tlwh(tlwh_to_cxcysr(a)), a, atol = 0.05).all())
    
    
def test_coordinates_transform_one_way():
    # Test bounding box -> KF state: [left, top, w, h] -> [xc, yc, s, r]
    bb = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    # Expected: xc = 10 + 15 = 25, yc = 20 + 20 = 40, s = 30 * 40 = 1200, r = 30 / 40 = 0.75
    expected_kf = np.array([25.0, 40.0, 1200.0, 0.75], dtype=float)
    assert np.isclose(tlwh_to_cxcysr(bb), expected_kf, atol=1e-5).all()

    # Test KF state -> bounding box: [xc, yc, s, r] -> [left, top, w, h]
    kf = np.array([25.0, 40.0, 1200.0, 0.75], dtype=float)
    # Expected: w = sqrt(1200 * 0.75) = 30, h = sqrt(1200 / 0.75) = 40, left = 25 - 15 = 10, top = 40 - 20 = 20
    expected_bb = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    assert np.isclose(cxcysr_to_tlwh(kf), expected_bb, atol=1e-5).all()