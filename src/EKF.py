import numpy as np

class EKF:

    def __init__(self):
        pass

    def predict(self, theta_prev, P_prev, Q_prev):
        self.theta_ = theta_prev
        self.P_ = P_prev + Q_prev
        
    def update(self, dLdtheta, R, stageLoss):
        self.R = R
        p = np.shape(dLdtheta)[1] #size of theta
        S = np.matmul(np.matmul(dLdtheta, self.P_), np.transpose(dLdtheta)) + R
        kalmanGain = np.matmul(np.matmul(self.P_, np.transpose(dLdtheta)), np.linalg.inv(S)) #Kalman Gain
        self.P = np.matmul((np.eye(p) - np.matmul(kalmanGain, dLdtheta)), self.P_) #Update P
        self.theta = self.theta_ - np.matmul(kalmanGain,stageLoss).flatten()
        return self.theta
    
    # def C_update(self, F, C, y, b):
    #     IFF = np.eye(F.shape[0]) - np.matmul(np.linalg.pinv(F), F)
    #     R = np.eye(4) * 0.00000001
    #     H = IFF @ C.T @ np.linalg.inv(self.R) @ C @ IFF
    #     H = np.linalg.pinv(H)
    #     HCR = H @ C.T @ np.linalg.inv(self.R)
    #     theta = HCR @ y + (np.eye(H.shape[0]) - HCR @ C) @ np.linalg.pinv(F) @ b

    #     return theta
    
    def C_update(self, F, C, b):
        print(F.shape)
        print(C.shape)
        print(b.shape)
        print(self.P.shape)
        print(self.theta.shape)
        G = self.P @ F.T @ np.linalg.pinv(C @ self.P @ F.T)
        print(G.shape)
        
        self.P = np.matmul((np.eye(self.P.shape[0]) - np.matmul(G, F)), self.P)
        theta = self.theta + np.matmul(G, (b - np.matmul(F, self.theta)))

        return theta

if __name__ == '__main__':
    theta_prev = np.array((1, 2, 3))
    P_prev = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
    Q_prev = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]) 
    dLdtheta = np.array((1, 2, 3)) 
    R = 0.001
    stageLoss = 0.1

    updateTheta = EKF()
    updateTheta.predict(theta_prev, P_prev, Q_prev)
    updateTheta.update(dLdtheta, R, stageLoss)
    print(updateTheta.theta)