import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import OCIL
import JinEnv
import copy
# ------------------------------ Set up dynamic system ------------------------------
project = "RobotArm"
mode = "All"
saveFlag = False

trails = 20
dir = 'examples/robotarm/data/'
demoFile = 'robotarm_OCIL_original_constrained.mat'
noise = 0.

# initial guess
data = sio.loadmat(dir+demoFile)
true_theta = data['true_parameter'].flatten()
sigma = 0.6
nn_seed = 100
np.random.seed(nn_seed)
random_number = np.random.random(len(true_theta))
for i in range(trails-1):
    random_number = np.vstack((random_number, np.random.random(len(true_theta))))

for i in range(trails):
    dynsys = JinEnv.RobotArm()
    dynsys.initDyn(g = 0)
    dynsys.initCost(wu = 0.01)
    system = OCIL.ImitationLearning(project, mode, dynsys, noise, dir, demoFile, saveFlag)

    initial_theta = copy.deepcopy(true_theta)
    for idx in range(len(true_theta)):
        initial_theta[idx] = true_theta[idx] + 2*sigma * (random_number[i,idx] - 0.5)*true_theta[idx]
    print('initial_theta = ', initial_theta)
    system.initialize_theta(initial_theta)
    system.set_iteration(5)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(8) * 0.00000000001
    Q = np.eye(8) * 0.
    R = np.eye(6) * 0.0000001

    system.initialize_EKF(P, Q, R)

    system.solve()

