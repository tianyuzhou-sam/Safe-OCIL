import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import OCIL
import JinEnv

# ------------------------------ Set up dynamic system ------------------------------
project = "RobotArm"
mode = "All"
saveFlag = False
dynsys = JinEnv.RobotArm()
dynsys.initDyn(g = 0)
dynsys.initCost(wu = 0.01)
noise = 0.1

dir = 'examples/robotarm/data/'
demoFile = 'robotarm_OCIL_original_constrained.mat'

system = OCIL.ImitationLearning(project, mode, dynsys, noise, dir, demoFile, saveFlag)

# initial guess
data = sio.loadmat(dir+demoFile)
true_theta = data['true_parameter'].flatten()
sigma = 0.25
nn_seed = 1
np.random.seed(nn_seed)
initial_theta = true_theta + 2*sigma * (np.random.random(len(true_theta))-0.5)*true_theta
print('initial_theta = ', initial_theta)
system.initialize_theta(initial_theta)

system.set_iteration(1)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(8) * 0.00000000001
Q = np.eye(8) * 0.
R = np.eye(6) * 0.0000001

system.initialize_EKF(P, Q, R)

system.solve()
