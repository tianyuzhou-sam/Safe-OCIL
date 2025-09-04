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
saveFlag = False
dynsys = JinEnv.CartPole()
# dynsys.initDyn(mc=0.5, mp=0.5, l=1)
dynsys.initDyn()
dynsys.initCost(wu = 0.1)
dynsys.initConstraints()
# dynsys.initConstraints()

dir = 'examples/cartpole/data/'
# demoFile = 'cartpole_demos_soft.mat'
demoFile = 'cartpole_original_constrained.mat'
noise = 0.1
alpha = 5*1e-2
beta = 5*1e-2
system = SafeOCIL.ImitationLearning(project, mode, dynsys, noise, alpha, beta, dir, demoFile, saveFlag)

# initial guess
data = sio.loadmat(dir+demoFile)
true_theta = data['true_parameter'].flatten()
sigma = 0.05
nn_seed = 100
np.random.seed(nn_seed)
initial_theta = true_theta + sigma * np.random.random(len(true_theta))
print('initial_theta = ', initial_theta)
system.initialize_theta(initial_theta)
system.set_iteration(1)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(9) * 0.00000000001
Q = np.eye(9) * 0.
R = np.eye(5) * 0.00000001

# P = np.eye(9) * 0.0000000001
# Q = np.eye(9) * 0.
# R = np.eye(5) * 0.00000001

system.initialize_EKF(P, Q, R)

system.solve()

