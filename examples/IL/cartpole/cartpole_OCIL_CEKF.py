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
project = "CartPole"
mode = "All"
saveFlag = False
dynsys = JinEnv.CartPole()
# dynsys.initDyn(mc=0.5, mp=0.5, l=1)
dynsys.initDyn()
dynsys.initCost(wu = 0.1)
# dynsys.initConstraints(max_u=10, max_x=inf)
dynsys.initConstraints(max_x=inf)

dir = 'examples/IL/cartpole/data/'
demoFile = 'cartpole_demos_soft.mat'
noise = 0.
gamma = 1e1
system = OCIL.ConstrainedEKF(project, mode, dynsys, noise, gamma, dir, demoFile, saveFlag)
system.set_sigma(0.)
system.set_iteration(1)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(8) * 0.000000001
Q = np.eye(8) * 0.
R = np.eye(4) * 0.00000001


# unconstrained
# P = np.eye(7) * 0.000000001
# Q = np.eye(7) * 0.
# R = np.eye(5) * 0.00000001

system.initialize_EKF(P, Q, R)

system.solve()

