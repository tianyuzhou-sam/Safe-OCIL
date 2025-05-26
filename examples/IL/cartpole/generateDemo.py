import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import OCIL
import JinEnv
import generateTraj
inf = 1e20

# ------------------------------ Set up dynamic system ------------------------------
saveFlag = False
dynsys = JinEnv.CartPole()
mc = 0.5
mp = 0.5
l = 1
wx = 1
wq = 6
wdx = 1
wdq = 1
gamma = 1e-2

dynsys.initDyn(mc=mc, mp=mp, l=l)
dynsys.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu = 0.1)
dynsys.initConstraints(max_u=10, max_x=inf)
dt = 0.1
horizon = 30
init_state = [0,0,0,0]
true_parameter = [mc,mp,l,wx,wq,wdx,wdq]
dir = 'examples/IL/cartpole/data/'
demoFile = 'cartpole_demos_soft.mat'

generateTraj.generateTraj(dynsys, init_state, dt, horizon, gamma,true_parameter, dir, demoFile, saveFlag)

