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
dynsys = JinEnv.Quadrotor()
Jx = 1
Jy = 1
Jz = 1
mass = 1
l = 0.4
wr = 1
wv = 1
wq = 5
ww = 1

gamma = 1e-2

dynsys.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=l, c=0.01)
dynsys.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)
dynsys.initConstraints(min_u=-5, max_u=5, max_r=inf)
dt = 0.1
horizon = 50
init_state = [5,5,0,0,0,0,0,0,0,1,0,0,0]
true_parameter = [Jx,Jy,Jz,mass,l,wr,wv,wq,ww]
dir = 'examples/IL/uav/data/'
demoFile = 'uav_demos.mat'

generateTraj.generateTraj(dynsys, init_state, dt, horizon, gamma, true_parameter, dir, demoFile, saveFlag)

