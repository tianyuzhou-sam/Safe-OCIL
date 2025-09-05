import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafePDP
import PDP
import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random

# --------------------------- load demonstration data ----------------------------------------
saveFlag = True
data = sio.loadmat('examples/cartpole/data/cartpole_original_constrained.mat')
dt = data['dt'][0]
demo = data['trajectories'][0][0]

# true_parameter = [load['mc'], load['mp'], load['l'], load['max_u'], load['max_x'], load['wx'], load['wq'], load['wdx'],
#                   load['wdq']]
true_parameter = [0.5,0.5,1,5,0.8,0.1,1,0.1,0.1]
print('true parameter', true_parameter)

# ----------------------------main learning procedure ----------------------
# initial guess
trails = 100
sigma = 0.025
nn_seed = 100
np.random.seed(nn_seed)
random_seed = np.random.random(len(true_parameter))-0.5
random_number = [0,0,0,0,0,0,0,0,0]
random_number[:3] = random_seed[:3]
random_number[3:5] = random_seed[7:]
random_number[5:] = random_seed[3:7]
init_parameter = true_parameter + 2*sigma * np.array(random_number)
for i in range(trails-1):
    random_seed = np.random.random(len(true_parameter))-0.5
    random_number = [0,0,0,0,0,0,0,0,0]
    random_number[:3] = random_seed[:3]
    random_number[3:5] = random_seed[7:]
    random_number[5:] = random_seed[3:7]
    init_parameter = np.vstack((init_parameter, true_parameter + 2*sigma * np.array(random_number)))

# print('initial parameter', init_parameter)
# learning rate and maximum iteration
lr = 0.8e-5
max_iter = 11

# To protect from the case where the trajectory is not differentiable, usually in such a case, our experience is that
# the output trajectory (i.e., the derivative of the trajectory) from auxiliary system would have spikes. This case
# is rarely happen, but when it happens, we simply let the current trajectory derivative equal to the one in previous
# iteration.

for runs in range(trails):
    # -----------------------------  Load environment -----------------------------------------
    env = JinEnv.CartPole()
    # env.initDyn()
    env.initDyn()
    env_dyn = env.X + dt * env.f
    # env.initCost(wx=load['wx'], wq=load['wq'], wdx=load['wdx'], wdq=load['wdq'], wu=0.1)
    # true_parameter = [load['mc'], load['mp'], load['l'], load['max_u'], load['max_x']]
    env.initCost(wu=0.1)
    env.initConstraints()
    # ----------------------------create tunable coc object-----------------------
    coc = SafePDP.COCsys()
    # pass the system to coc
    coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar, env.cost_auxvar))
    print(coc.auxvar)
    coc.setStateVariable(env.X)
    coc.setControlVariable(env.U)
    coc.setDyn(env_dyn)
    # pass cost to coc
    coc.setPathCost(env.path_cost)
    coc.setFinalCost(env.final_cost)
    # pass constraints to coc
    coc.setPathInequCstr(env.path_inequ)
    # differentiating CPMP
    coc.diffCPMP()
    # convert to the unconstrained barrier OC object
    gamma = 1e-2
    coc.convert2BarrierOC(gamma=gamma)

    # initialize the storage
    loss_trace_barrierOC = []  # use theorem 2 to approximate both the system trajectory and its derivative
    parameter_trace_barrierOC = np.empty((max_iter, coc.n_auxvar))

    grad_protection_threshold = 1e5
    previous_grad_barrierOC = 0

    current_parameter_barrierOC = init_parameter[runs]
    print('current initial parameter', current_parameter_barrierOC)

    for k in range(max_iter):
        loss_barrierOC = 0
        grad_barrierOC = 0

        # fetch the data sample
        init_state = demo['state_traj_opt'][0, :]
        horizon = demo['control_traj_opt'].shape[0]

        # Strategy 2：
        # use theorem 2 to approximate both the system trajectory and its derivative
        traj_barrierOC = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                            auxvar_value=current_parameter_barrierOC)
        aux_sol_barrierOC = coc.auxSysBarrierOC(opt_sol=traj_barrierOC)
        loss_barrierOC, grad_barrierOC = SafePDP.Traj_L2_Loss(demo, traj_barrierOC, aux_sol_barrierOC)

        # protect the non-differentiable case for Strategy 2
        if norm_2(grad_barrierOC) > grad_protection_threshold:
            grad_barrierOC = previous_grad_barrierOC
        else:
            previous_grad_barrierOC = grad_barrierOC

        # storage
        loss_trace_barrierOC += [loss_barrierOC]
        parameter_trace_barrierOC[k] = current_parameter_barrierOC

        # print
        np.set_printoptions(suppress=True)
        print('iter #:', k, ' loss_barrierOC:', loss_barrierOC)

        # update
        current_parameter_barrierOC = current_parameter_barrierOC - lr * grad_barrierOC

    # save
    if saveFlag:
        # save_data = {'parameter_trace_barrierOC': parameter_trace_barrierOC,
        #              'loss_trace_barrierOC': loss_trace_barrierOC,
        #              'gamma': gamma,
        #              'nn_seed': nn_seed,
        #              'lr': lr,
        #              'init_parameter': init_parameter,
        #              'true_parameter': true_parameter}
        # np.save('Examples/MPC/Results/CIOC_Cartpole_trial_1.npy', save_data)

        sio.savemat("examples/cartpole/data/results/results_" + str(runs+1) + ".mat", {'parameter_trace_barrierOC': parameter_trace_barrierOC,
                    'loss_trace_barrierOC': loss_trace_barrierOC,
                    'gamma': gamma,
                    'nn_seed': nn_seed,
                    'lr': lr,
                    'init_parameter': init_parameter,
                    'true_parameter': true_parameter})

