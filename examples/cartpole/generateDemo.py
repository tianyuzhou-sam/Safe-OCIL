import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv
import generateTraj

# ------------------------------ Set up dynamic system ------------------------------
saveFlag = False
dynsys = JinEnv.CartPole()
mc = 0.5
mp = 0.5
l = 1
wx = 0.1
wq = 1
wdx = 0.1
wdq = 0.1
max_u = 5
max_x = 0.8
alpha = 4*7.5*1e-2
beta = 7.5*1e-2

dynsys.initDyn(mc=mc, mp=mp, l=l)
dynsys.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu = 0.1)
dynsys.initConstraints(max_u=max_u, max_x=max_x)

dt = 0.1
horizon = 35
init_state = [0,0,0,0]
true_parameter = [mc,mp,l,wx,wq,wdx,wdq,max_u,max_x]
dir = 'examples/cartpole/data/'
demoFile = 'cartpole'

traj, coctraj = generateTraj.generateTraj().generateTraj(dynsys, init_state, dt, horizon, alpha, beta, true_parameter, dir, demoFile, saveFlag)

dynsys.play_animation(pole_len=2, dt=dt, state_traj=coctraj['state_traj_opt'])

dynsys.play_animation(pole_len=2, dt=dt, state_traj=traj['state_traj_opt'])
