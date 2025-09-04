import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv
inf = 1e20

# ------------------------------ Set up dynamic system ------------------------------
project = "CartPole"
mode = "All"
saveFlag = True


dir = 'examples/cartpole/data/'
demoFile = 'cartpole_original_constrained.mat'
noise = 0.
alpha = 5*1e-2
beta = 5*1e-2

# initial guess
data = sio.loadmat(dir+demoFile)
true_theta = data['true_parameter'].flatten()
sigma = 0.05
nn_seed = 100
np.random.seed(nn_seed)

for i in range(100):
    dynsys = JinEnv.CartPole()
    dynsys.initDyn()
    dynsys.initCost(wu = 0.1)
    dynsys.initConstraints()
    system = SafeOCIL.ImitationLearning(project, mode, dynsys, noise, alpha, beta, dir, demoFile, saveFlag)

    initial_theta = true_theta + sigma * np.random.random(len(true_theta))
    print('initial_theta = ', initial_theta)
    system.initialize_theta(initial_theta)
    system.set_iteration(5)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(9) * 0.00000000001
    Q = np.eye(9) * 0.
    R = np.eye(5) * 0.00000001
    

    system.initialize_EKF(P, Q, R)

    system.solve()

