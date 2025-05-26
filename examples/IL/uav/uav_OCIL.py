import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import OCIL
import JinEnv
inf = 1e20

# ------------------------------ Set up dynamic system ------------------------------
project = "Quadrotor"
mode = "All"
saveFlag = False
dynsys = JinEnv.Quadrotor()
dynsys.initDyn(c=0.01)
dynsys.initCost(wthrust=0.1)
dynsys.initConstraints(min_u=-5, max_u=5, max_r=inf)

dir = 'examples/IL/uav/data/'
demoFile = 'uav_demos.mat'
noise = 0.0
gamma = 1e-2

system = OCIL.ImitationLearning(project, mode, dynsys, noise, gamma, dir, demoFile, saveFlag)
system.set_sigma(0.2)
system.set_iteration(1)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(9) * 0.0000001
Q = np.eye(9) * 0.
R = np.eye(17) * 0.0000000001

system.initialize_EKF(P, Q, R)

system.solve()