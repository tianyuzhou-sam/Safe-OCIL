import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv
import generateTraj
import generateTrajAlphaBeta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ocSolver

# ------------------------------ Set up dynamic system ------------------------------
visualize_only = True

if not visualize_only:
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
    alpha = 0.05
    beta = 0.05
    dt = 0.1


    true_parameter = [mc,mp,l,wx,wq,wdx,wdq,max_u,max_x]
    trails = 1
    sigma = 0.1
    nn_seed = 1
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

    dynsys = JinEnv.CartPole()
    dynsys.initDyn(mc=true_parameter[0], mp=true_parameter[1], l=true_parameter[2])
    dynsys.initCost(wx=true_parameter[3], wq=true_parameter[4], wdx=true_parameter[5], wdq=true_parameter[6], wu = 0.1)
    dynsys.initConstraints(max_u=true_parameter[7], max_x=true_parameter[8])

    horizon = 35
    init_state = [0,0,0,0]
    dir = 'examples/cartpole/data/'
    demoFile = 'cartpole'

    traj, coctraj = generateTraj.generateTraj().generateTraj(dynsys, init_state, dt, horizon, alpha, beta, true_parameter, dir, demoFile, saveFlag)

    demo_state_traj = coctraj['state_traj_opt']
    demo_control_traj = coctraj['control_traj_opt']
    demo_traj_list = np.hstack((demo_state_traj.reshape(-1), demo_control_traj.reshape(-1)))
        
    True_sysoc = ocSolver.OCSys()
    True_sys = JinEnv.CartPole()
    True_sys.initDyn()
    True_sys.initCost(wu=0.1)
    True_sys.initConstraints()
    True_sysoc.setAuxvarVariable(vertcat(True_sys.dyn_auxvar, True_sys.cost_auxvar, True_sys.constraint_auxvar))
    True_sysoc.setControlVariable(True_sys.U)
    True_sysoc.setStateVariable(True_sys.X)
    True_dyn = True_sys.X + dt * True_sys.f
    True_sysoc.setDyn(True_dyn)
    True_sysoc.setPathCost(True_sys.path_cost)
    True_sysoc.setFinalCost(True_sys.final_cost)
    True_sysoc.setPathInequCstr(True_sys.path_inequ)
    True_sysoc.diffCPMP()

    traj, coctraj = generateTraj.generateTraj().generateTraj(dynsys, init_state, dt, horizon, alpha, beta, true_parameter, dir, demoFile, saveFlag)


    True_clqr = ocSolver.EQCLQR()
    True_auxsys = True_sysoc.getAuxSys(opt_sol=coctraj, threshold=1e-2)
    True_clqr.auxsys2Eqctlqr(auxsys=True_auxsys)
    True_aux_sol = True_clqr.eqctlqrSolver(threshold=1e-2)
    True_dxdtheta_traj = np.array(True_aux_sol['state_traj_opt'])
    True_dudtheta_traj = np.array(True_aux_sol['control_traj_opt'])
    demo_dxidtheta_list = np.hstack((True_dxdtheta_traj.reshape(-1), True_dudtheta_traj.reshape(-1)))
    # demo_dxidtheta_list = None


    alpha_list = np.linspace(0.05, 0.5, 30)
    alpha_list = alpha_list.tolist()
    beta_list = np.linspace(0.05, 0.5, 30)
    beta_list = beta_list.tolist()
    L = np.zeros((trails, len(alpha_list), len(beta_list)))
    T = np.zeros((trails, len(alpha_list), len(beta_list)))
    L_dxidtheta = np.zeros((trails, len(alpha_list), len(beta_list)))
    dxidtheta = np.zeros((trails, len(alpha_list), len(beta_list), len(demo_dxidtheta_list)))


    for i in range(trails):
        if trails == 1:
            parameter = true_parameter
        else:
            parameter = init_parameter[i]

        for alpha in alpha_list:
            for beta in beta_list:

                traj, time = generateTrajAlphaBeta.generateTrajAlphaBeta().generateTraj(dynsys, init_state, dt, horizon, alpha, beta, parameter, dir, demoFile, saveFlag)
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']


                sysoc = ocSolver.OCSys()
                sys = JinEnv.CartPole()
                sys.initDyn()
                sys.initCost(wu=0.1)
                sys.initConstraints()
                sysoc.setAuxvarVariable(vertcat(sys.dyn_auxvar, sys.cost_auxvar, sys.constraint_auxvar))
                sysoc.setControlVariable(sys.U)
                sysoc.setStateVariable(sys.X)
                dyn = sys.X + dt * sys.f
                sysoc.setDyn(dyn)
                sysoc.setPathCost(sys.path_cost)
                sysoc.setFinalCost(sys.final_cost)
                sysoc.setPathInequCstr(sys.path_inequ)
                sysoc.diffCPMP()

                clqr = ocSolver.EQCLQR()
                sysoc.convert2BarrierOC(alpha=alpha, beta=beta)

                traj = sysoc.solveBarrierOC(ini_state=init_state, horizon=horizon, auxvar_value = parameter)
                aux_sol = sysoc.auxSysBarrierOC(opt_sol=traj)

                if demo_dxidtheta_list is None:
                    demo_dxidtheta_list = np.hstack((np.array(aux_sol['state_traj_opt']).reshape(-1), np.array(aux_sol['control_traj_opt']).reshape(-1)))

                dxdtheta_traj = np.array(aux_sol['state_traj_opt'])
                dudtheta_traj = np.array(aux_sol['control_traj_opt'])

                
                Loss = 0
                percent = 0
                for jdx in range(horizon):
                    lossNorm = norm_2(state_traj[jdx]-demo_state_traj[jdx])**2 + norm_2(control_traj[jdx]-demo_control_traj[jdx])**2
                    Loss += lossNorm

                
                traj_list = np.hstack((state_traj.reshape(-1), control_traj.reshape(-1)))
                dxidtheta_list = np.hstack((dxdtheta_traj.reshape(-1), dudtheta_traj.reshape(-1)))
                

                percent = norm_2(demo_traj_list - traj_list)/norm_2(demo_traj_list)
                # percent_dxidtheta = norm_2(demo_dxidtheta_list - dxidtheta_list)/norm_2(demo_dxidtheta_list)

                
                
                T[i, alpha_list.index(alpha), beta_list.index(beta)] = time
                L[i, alpha_list.index(alpha), beta_list.index(beta)] = percent
                # L_dxidtheta[i, alpha_list.index(alpha), beta_list.index(beta)] = percent_dxidtheta
                dxidtheta[i, alpha_list.index(alpha), beta_list.index(beta)] = dxidtheta_list
                
                print(f"iter {i}, Alpha: {alpha}, Beta: {beta}, , loss: {Loss}, loss_percent: {percent}, Time: {time}")
                

    sio.savemat('examples/cartpole/data/results/compare_parameter.mat', {'alpha_list': alpha_list, 'beta_list': beta_list, 'L': L, 'dxidtheta': dxidtheta, 'T': T})

if visualize_only:
    data = sio.loadmat('examples/cartpole/data/results/compare_parameter_traj.mat')
else:
    data = sio.loadmat('examples/cartpole/data/results/compare_parameter.mat')
alpha_list = data['alpha_list']
alpha_list = alpha_list.flatten().tolist()
beta_list = data['beta_list']
beta_list = beta_list.flatten()
L = data['L']
T = data['T']
dxidtheta = data['dxidtheta']
dxidtheta = dxidtheta[0]

L_avg = np.mean(L, axis=0)

print(L_avg)
print(alpha_list)
print(beta_list)

# Find minimum L_avg and corresponding alpha, beta
min_idx = np.unravel_index(np.argmin(L_avg), L_avg.shape)
min_alpha = alpha_list[min_idx[0]]
min_beta = beta_list[min_idx[1]]
min_loss = L_avg[min_idx]
print(f"\nMinimum Loss: {min_loss}")
print(f"Optimal Alpha: {min_alpha}")
print(f"Optimal Beta: {min_beta}")
print(f"Indices: alpha_idx={min_idx[0]}, beta_idx={min_idx[1]}")


dxidtheta_min = dxidtheta[min_idx]
dxidtheta_error = np.zeros((len(alpha_list), len(beta_list)))
for i in range(len(alpha_list)):
    for j in range(len(beta_list)):
        dxidtheta_error[i, j] = norm_2(dxidtheta[i, j] - dxidtheta_min)/norm_2(dxidtheta_min)



# fig, axs = plt.subplots()
# for alpha in alpha_list:
#     axs.plot(beta_list, L_avg[alpha_list.index(alpha)], label=f'Alpha: {alpha}')
# axs.set_xlabel('Beta')
# axs.set_ylabel('Loss')
# axs.set_title('Loss')
# axs.set_yscale('log')
# axs.legend()

# 3D surface plot
fig_3d = plt.figure(figsize=(9, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Create meshgrid for alpha and beta
Alpha, Beta = np.meshgrid(alpha_list, beta_list, indexing='ij')

# Plot surface
surf = ax_3d.plot_surface(Alpha, Beta, L_avg, cmap='viridis', alpha=0.8, edgecolor='none')
ax_3d.set_xlabel(r'$\alpha$', fontsize=16)
ax_3d.set_ylabel(r'$\beta$', fontsize=16)
ax_3d.set_zlabel('Estimation Error', fontsize=16, labelpad=10)
ax_3d.tick_params(axis='x', labelsize=16)
ax_3d.tick_params(axis='y', labelsize=16)
ax_3d.tick_params(axis='z', labelsize=16)
# ax_3d.set_title('Loss vs Alpha and Beta', fontsize=14)
# ax_3d.set_zscale('log')
# ax_3d.set_zlim(0, 1)

# Mark the minimum point
ax_3d.scatter([min_alpha], [min_beta], [min_loss], color='red', s=300, marker='*', 
              label=f'Minimum: $\\alpha$={min_alpha:.4f}, $\\beta$={min_beta:.4f}')
ax_3d.legend(fontsize=16)

# Add colorbar
fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=20, pad=0.05)

# Original 3D surface plot for L_dxidtheta_avg (all data)
fig_3d_dxidtheta = plt.figure(figsize=(9, 7))
ax_3d_dxidtheta = fig_3d_dxidtheta.add_subplot(111, projection='3d')

# Create meshgrid for alpha and beta
Alpha_dxidtheta, Beta_dxidtheta = np.meshgrid(alpha_list, beta_list, indexing='ij')

# Plot surface (all data)
surf_dxidtheta = ax_3d_dxidtheta.plot_surface(Alpha_dxidtheta, Beta_dxidtheta, dxidtheta_error, 
                                               cmap='viridis', alpha=0.8, edgecolor='none')
ax_3d_dxidtheta.set_xlabel(r'$\alpha$', fontsize=16)
ax_3d_dxidtheta.set_ylabel(r'$\beta$', fontsize=16)
ax_3d_dxidtheta.set_zlabel('Gradient Estimation Error', fontsize=16, labelpad=10)
ax_3d_dxidtheta.tick_params(axis='x', labelsize=16)
ax_3d_dxidtheta.tick_params(axis='y', labelsize=16)
ax_3d_dxidtheta.tick_params(axis='z', labelsize=16)


# Add colorbar
fig_3d_dxidtheta.colorbar(surf_dxidtheta, ax=ax_3d_dxidtheta, shrink=0.5, aspect=20, pad=0.05)
fig_3d_dxidtheta.tight_layout()

# New 3D surface plot for L_dxidtheta_avg (only alpha >= beta)
fig_3d_dxidtheta_masked = plt.figure(figsize=(9, 7))
ax_3d_dxidtheta_masked = fig_3d_dxidtheta_masked.add_subplot(111, projection='3d')

# Create meshgrid for alpha and beta
Alpha_dxidtheta_masked, Beta_dxidtheta_masked = np.meshgrid(alpha_list, beta_list, indexing='ij')

# Mask data where alpha < beta - 0.05
L_dxidtheta_masked = dxidtheta_error.copy()
mask = Alpha_dxidtheta_masked < (Beta_dxidtheta_masked + 0.02)
L_dxidtheta_masked[mask] = np.nan

# Plot surface (only where alpha >= beta - 0.05)
surf_dxidtheta_masked = ax_3d_dxidtheta_masked.plot_surface(Alpha_dxidtheta_masked, Beta_dxidtheta_masked, L_dxidtheta_masked, 
                                               cmap='viridis', alpha=0.8, edgecolor='none')
ax_3d_dxidtheta_masked.set_xlabel(r'$\alpha$', fontsize=16)
ax_3d_dxidtheta_masked.set_ylabel(r'$\beta$', fontsize=16)
ax_3d_dxidtheta_masked.set_zlabel('Gradient Estimation Error', fontsize=16, labelpad=10)
ax_3d_dxidtheta_masked.tick_params(axis='x', labelsize=16)
ax_3d_dxidtheta_masked.tick_params(axis='y', labelsize=16)
ax_3d_dxidtheta_masked.tick_params(axis='z', labelsize=16)

# Add colorbar
fig_3d_dxidtheta_masked.colorbar(surf_dxidtheta_masked, ax=ax_3d_dxidtheta_masked, shrink=0.5, aspect=20, pad=0.05)
fig_3d_dxidtheta_masked.tight_layout()

fig_3d.tight_layout()

plt.show()







