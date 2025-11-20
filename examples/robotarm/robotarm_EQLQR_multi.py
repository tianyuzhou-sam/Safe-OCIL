import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL_EQLQR
import JinEnv
import copy
inf = 1e20

# ------------------------------ Set up dynamic system ------------------------------
project = "RobotArm"
mode = "All"
saveFlag = False

trails = 100
dir = 'examples/robotarm/data/'
demoFile = 'robotarm_original_constrained.mat'
noise = 0.
alpha = 10*1e-2
beta = 10*1e-2

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
    dynsys.initConstraints()
    system = SafeOCIL_EQLQR.ImitationLearning(project, mode, dynsys, noise, alpha, beta, dir, demoFile, saveFlag)

    initial_theta = copy.deepcopy(true_theta)
    for idx in range(len(true_theta)):
        initial_theta[idx] = true_theta[idx] + 2*sigma * (random_number[i,idx] - 0.5)*true_theta[idx]
    print('initial_theta = ', initial_theta)
    system.initialize_theta(initial_theta)
    system.set_iteration(5)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(10) * 0.000000001
    Q = np.eye(10) * 0.
    R = np.eye(6) * 0.00000001

    system.initialize_EKF(P, Q, R)

    system.solve()

