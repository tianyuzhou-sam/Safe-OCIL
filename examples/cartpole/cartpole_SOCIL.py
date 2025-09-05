import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv

# ------------------------------ Set up dynamic system ------------------------------
project = "CartPole"
mode = "All"
saveFlag = False
dynsys = JinEnv.CartPole()
dynsys.initDyn()
dynsys.initCost(wu = 0.1)
dynsys.initConstraints()

dir = 'examples/cartpole/data/'
demoFile = 'cartpole_original_constrained.mat'
noise = 0.2
alpha = 4*7.5*1e-2
beta = 7.5*1e-2
system = SafeOCIL.ImitationLearning(project, mode, dynsys, noise, alpha, beta, dir, demoFile, saveFlag)

# initial guess
data = sio.loadmat(dir+demoFile)
true_theta = data['true_parameter'].flatten()
sigma = 0.1
nn_seed = 1
np.random.seed(nn_seed)
initial_theta = true_theta + 2*sigma * (np.random.random(len(true_theta))-0.5)
print('initial_theta = ', initial_theta)
system.initialize_theta(initial_theta)
system.set_iteration(1)

# --------------------------- initilize EKF ----------------------------------------
# no noise
P = np.eye(9) * 0.00000000001
Q = np.eye(9) * 0.
R = np.eye(5) * 0.00000001

# 0.2 noise
P = np.eye(9) * 0.000000001
Q = np.eye(9) * 0.
R = np.eye(5) * 0.00000001

system.initialize_EKF(P, Q, R)

system.solve()

