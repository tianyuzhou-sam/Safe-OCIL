import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv
import matplotlib.pyplot as plt
# ------------------------------ Set up dynamic system ------------------------------
visualize_only = True
if not visualize_only:
    project = "CartPole"
    mode = "All"
    saveFlag = False

    trails = 10
    dir = 'examples/cartpole/data/'
    demoFile = 'cartpole_original_constrained.mat'
    noise = 0.
    alpha = 4*7.5*1e-2
    beta = 7.5*1e-2

    # initial guess
    data = sio.loadmat(dir+demoFile)
    true_theta = data['true_parameter'].flatten()
    sigma = 0.1
    nn_seed = 1
    np.random.seed(nn_seed)
    random_number = np.random.random(len(true_theta))
    for i in range(trails-1):
        random_number = np.vstack((random_number, np.random.random(len(true_theta))))


    alpha_list = np.linspace(0.05, 0.5, 30)
    alpha_list = alpha_list.tolist()
    beta_list = np.linspace(0.05, 0.5, 30)
    beta_list = beta_list.tolist()
    L = np.zeros((len(alpha_list), len(beta_list)))
    L_min = np.zeros((len(alpha_list), len(beta_list)))

    for alpha in alpha_list:
        for beta in beta_list:
            loss_this_parameter = np.zeros(trails)
            for n_trail in range(trails):
                dynsys = JinEnv.CartPole()
                dynsys.initDyn()
                dynsys.initCost(wu = 0.1)
                dynsys.initConstraints()
                system = SafeOCIL.ImitationLearning(project, mode, dynsys, noise, alpha, beta, dir, demoFile, saveFlag)

                initial_theta = true_theta + 2*sigma * (random_number[n_trail] - 0.5)
                print('true_theta = ', true_theta)
                print('initial_theta = ', initial_theta)
                system.initialize_theta(initial_theta)
                system.set_iteration(1)

                # --------------------------- initilize EKF ----------------------------------------
                # # no noise
                P = np.eye(9) * 0.00000000001
                Q = np.eye(9) * 0.
                R = np.eye(5) * 0.00000001
                
                system.initialize_EKF(P, Q, R)

                Loss_his = system.solve()
                loss_this_parameter[n_trail] = Loss_his[-1]
                L_min[alpha_list.index(alpha), beta_list.index(beta)] = np.min(Loss_his)

            L[alpha_list.index(alpha), beta_list.index(beta)] = np.mean(loss_this_parameter)
            print(f"alpha: {alpha}, beta: {beta}, mean loss: {L[alpha_list.index(alpha), beta_list.index(beta)]}")

    sio.savemat('examples/cartpole/data/results/compare_parameter_loss.mat', {'alpha_list': alpha_list, 'beta_list': beta_list, 'L': L, 'L_min': L_min})

data = sio.loadmat('examples/cartpole/data/results/compare_parameter_loss.mat')
alpha_list = data['alpha_list']
alpha_list = alpha_list.flatten().tolist()
beta_list = data['beta_list']
beta_list = beta_list.flatten()
L = data['L']
L_min = data['L_min']

print(L)

for i in range(len(alpha_list)):
    for j in range(len(beta_list)):
        if L[i, j] >10:
            L[i, j] = np.inf

min_idx = np.unravel_index(np.argmin(L), L.shape)
min_alpha = alpha_list[min_idx[0]]
min_beta = beta_list[min_idx[1]]
min_loss = L[min_idx]

min_idx_min = np.unravel_index(np.argmin(L_min), L_min.shape)
min_alpha_min = alpha_list[min_idx_min[0]]
min_beta_min = beta_list[min_idx_min[1]]
min_loss_min = L_min[min_idx_min]

# 3D surface plot
fig_3d = plt.figure(figsize=(9, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Create meshgrid for alpha and beta
Alpha, Beta = np.meshgrid(alpha_list, beta_list, indexing='ij')

# Plot surface
surf = ax_3d.plot_surface(Alpha, Beta, L, cmap='viridis', alpha=0.8, edgecolor='none')
ax_3d.set_xlabel(r'$\alpha$', fontsize=16)
ax_3d.set_ylabel(r'$\beta$', fontsize=16)
ax_3d.set_zlabel('Loss', fontsize=16, labelpad=0)
# Rotate z-axis label to match y-axis direction (horizontal)
ax_3d.zaxis.label.set_rotation(90)
ax_3d.tick_params(axis='x', labelsize=16)
ax_3d.tick_params(axis='y', labelsize=16)
ax_3d.tick_params(axis='z', labelsize=16)
# ax_3d.set_title('Loss vs Alpha and Beta', fontsize=14)
# ax_3d.set_zscale('log')
# ax_3d.set_zlim(1e0, 1e3)

# Mark the minimum point
ax_3d.scatter([min_alpha], [min_beta], [min_loss], color='red', s=300, marker='*', 
              label=f'Minimum: $\\alpha$={min_alpha:.4f}, $\\beta$={min_beta:.4f}')
ax_3d.legend(fontsize=16)

# Add colorbar
fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=20, pad=0.05)
ax_3d.legend(fontsize=16)
plt.tight_layout()

plt.show()