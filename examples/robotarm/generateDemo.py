import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv
import generateTraj
inf = 1e20

# ------------------------------ Set up dynamic system ------------------------------
saveFlag = False
dynsys = JinEnv.RobotArm()
m1, m2, l1, l2 = 1, 1, 1, 1
dynsys.initDyn(m1=m1, m2=m2, l1=l1, l2=l2, g=0)
wq1, wq2, wdq1, wdq2, wu = 0.1, 0.1, 0.1, 0.1, 0.01
dynsys.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu)
max_u = 1
max_q = pi
dynsys.initConstraints(max_u=max_u, max_q=max_q)

alpha = 5*4*1e-3
beta = 5*1e-3
dt = 0.2
horizon = 25
init_state = [-pi / 2, 3 * pi / 4, 0, 0]
true_parameter = [m1,m2,l1,l2,wq1,wq2,wdq1,wdq2,max_u,max_q]
dir = 'examples/robotarm/data/'
demoFile = 'robotarm'

traj, coctraj = generateTraj.generateTraj().generateTraj(dynsys, init_state, dt, horizon, alpha, beta, true_parameter, dir, demoFile, saveFlag)

dynsys.play_animation(l1=1, l2=1, dt=dt, state_traj=coctraj['state_traj_opt'])

dynsys.play_animation(l1=1, l2=1, dt=dt, state_traj=traj['state_traj_opt'])
