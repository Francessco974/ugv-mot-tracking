# mot_tracking/kalman.py
import numpy as np

class SortKF:
    """
    State: [u, v, s, r, u_dot, v_dot, s_dot]
    u → x-coord center
    v → y-coord center
    s → Area
    r → Aspect ratio 
    """
    
    def __init__(self, dt = 1):
        
        self.dt = dt
        self.state_ = np.zeros((7,))
        self.P_ = np.eye(7)
        self.Q_ = np.eye(7)
        self.R_ = np.eye(4)
        # from SORT article
        self.R_[2:  , 2: ]*=10
        self.Q_[-1: , -1:]*=0.01
        self.Q_[4:  , 4: ]*=0.01        
        
        self.S_ = np.eye(4)
        self.I_ = np.eye(7)
        
        self.F_ = np.array([[1, 0, 0, 0, dt, 0,  0],
                           [0, 1, 0, 0, 0,  dt, 0],
                           [0, 0, 1, 0, 0,  0,  dt],
                           [0, 0, 0, 1, 0,  0,  0],
                           [0, 0, 0, 0, 1,  0,  0],
                           [0, 0, 0, 0, 0,  1,  0],
                           [0, 0, 0, 0, 0,  0,  1]
                           ])
        
        # You measure xc, yc, area, ratio from the bounding boxes
        self.H_ = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]
                           ])
        
     
        
    def predict(self):
        """
        Predict next step
        """
        
        # Guard scale to become negative
        if (self.state_[2] + self.dt*self.state_[6] <= 0):
            self.state_[6] = 0.0
            
        self.state_ = self.F_ @ self.state_
        self.P_ = self.F_ @ self.P_ @ self.F_.T + self.Q_
        return self.state_[:4]
    
    def update(self, measurement : np.ndarray):
        """ 
        measurement: np.array([xc, yc, Area, r])
        """
        
        if measurement.size != 4:
            raise ValueError(f"Measurement shape does not match requirement (expected 4 elements, got shape {measurement.shape})")
        
        y = measurement - self.H_ @ self.state_
        S = self.H_ @ self.P_ @ self.H_.T + self.R_ 
        # It is equal to K = P H^T S^{-1} but more numerically stable
        K = np.linalg.solve(S, self.H_ @ self.P_).T
        
        self.state_ = self.state_ + K @ y
        
        # Joseph Form Update
        temp_matrix_ = (self.I_ - K @ self.H_)
        self.P_ = temp_matrix_ @ self.P_ @ temp_matrix_.T + K @ self.R_ @ K.T
        self.S_ = S
         
    def set_initial_state_(self, state):
        self.state_ = state
    
    def initiate(self, measurement : np.ndarray, P0 : np.ndarray):
        
        if measurement.size != 4:
            raise ValueError(f"Measurement shape does not match requirement (expected 4 elements, got shape {measurement.shape})")
                
        if P0.shape != (7 ,7):
            raise ValueError(f"Measurement shape does not match requirement (expected (7 x 7) elements, got shape {P0.shape})")
        
        self.state_ = np.array([measurement[0], measurement[1], measurement[2], measurement[3], 0, 0, 0], dtype=float)
        self.P_ = P0

    
    
    
    

