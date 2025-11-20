import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

params = {'axes.labelsize': 28,
          'axes.titlesize': 28,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 20}
plt.rcParams.update(params)

# Load the results file
trails = 100
for i in range(trails):
    results_file = f'examples/cartpole/data/results/SOCIL/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_SOCIL = results['Loss']
        SOCIL_time = results['SOCIL_time']
        estimator_time = results['Estimator_time']
        gradient_time = results['Gradient_time']
    else:
        Loss_SOCIL = np.vstack((Loss_SOCIL, results['Loss']))
        SOCIL_time = np.vstack((SOCIL_time, results['SOCIL_time']))
        estimator_time = np.vstack((estimator_time, results['Estimator_time']))
        gradient_time = np.vstack((gradient_time, results['Gradient_time']))

horizon = len(results['control'][0])

OCIL_trails = 20
for i in range(OCIL_trails):
    results_file = f'examples/cartpole/data/results/OCIL/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        OCIL_time = results['OCIL_time']
        OCIL_gradient_time = results['Gradient_time']
    else:
        OCIL_time = np.vstack((OCIL_time, results['OCIL_time']))
        OCIL_gradient_time = np.vstack((OCIL_gradient_time, results['Gradient_time']))


print('SOCIL_time = ', np.mean(SOCIL_time)*1000)
print('SOCIL STD = ', np.std(SOCIL_time)*1000)
print('gradient_time = ', np.mean(gradient_time)*1000)
print('gradient STD = ', np.std(gradient_time)*1000)
print('estimator_time = ', np.mean(estimator_time)*1000)
print('estimator STD = ', np.std(estimator_time)*1000)

print('OCIL_time = ', np.mean(OCIL_time)*1000)
print('OCIL STD = ', np.std(OCIL_time)*1000)
print('OCIL gradient_time = ', np.mean(OCIL_gradient_time)*1000)
print('OCIL gradient STD = ', np.std(OCIL_gradient_time)*1000)

print('--------------------------------')

SOCIL_Loss_avg = np.mean(Loss_SOCIL, axis=0)
SOCIL_Loss_std = np.std(Loss_SOCIL, axis=0)
SOCIL_Loss_ub = SOCIL_Loss_avg + 3*SOCIL_Loss_std
SOCIL_Loss_lb = SOCIL_Loss_avg - 3*SOCIL_Loss_std
t_SOCIL = np.arange(0, len(Loss_SOCIL[0]))

for i in range(1,trails):
    results_file = f'examples/cartpole/data/results/SPDP/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 1:
        Loss_SPDP = results['loss_trace_barrierOC']
    else:
        # remove bad runs from SPDP
        if np.mean(results['loss_trace_barrierOC'][0]) < 1000:
            Loss_SPDP = np.vstack((Loss_SPDP, results['loss_trace_barrierOC']))


SPDP_Loss_avg = np.mean(Loss_SPDP, axis=0)
SPDP_Loss_std = np.std(Loss_SPDP, axis=0)
SPDP_Loss_ub = SPDP_Loss_avg + 3*SPDP_Loss_std
SPDP_Loss_lb = SPDP_Loss_avg - 3*SPDP_Loss_std
t_SPDP = list(range(0, len(Loss_SPDP[0])))
t_SPDP = [x*horizon for x in t_SPDP]

SOCIL_results_file = 'examples/cartpole/data/results/SOCIL.mat'
SOCIL_results = sio.loadmat(SOCIL_results_file)
OCIL_results_file = 'examples/cartpole/data/results/OCIL.mat'
OCIL_results = sio.loadmat(OCIL_results_file)






trails = 100
for i in range(trails):
    results_file = f'examples/robotarm/data/results/SOCIL/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_SOCIL_robotarm = results['Loss']
        SOCIL_time_robotarm = results['SOCIL_time']
        estimator_time_robotarm = results['Estimator_time']
        gradient_time_robotarm = results['Gradient_time']
    else:
        Loss_SOCIL_robotarm = np.vstack((Loss_SOCIL_robotarm, results['Loss']))
        SOCIL_time_robotarm = np.vstack((SOCIL_time_robotarm, results['SOCIL_time']))
        estimator_time_robotarm = np.vstack((estimator_time_robotarm, results['Estimator_time']))
        gradient_time_robotarm = np.vstack((gradient_time_robotarm, results['Gradient_time']))

horizon_robotarm = len(results['control'][0])

OCIL_trails = 20
for i in range(OCIL_trails):
    results_file = f'examples/robotarm/data/results/OCIL/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        OCIL_time_robotarm = results['OCIL_time']
        OCIL_gradient_time_robotarm = results['Gradient_time']
    else:
        OCIL_time_robotarm = np.vstack((OCIL_time_robotarm, results['OCIL_time']))
        OCIL_gradient_time_robotarm = np.vstack((OCIL_gradient_time_robotarm, results['Gradient_time']))


print('SOCIL_time_robotarm = ', np.mean(SOCIL_time_robotarm)*1000)
print('SOCIL STD_robotarm = ', np.std(SOCIL_time_robotarm)*1000)
print('gradient_time_robotarm = ', np.mean(gradient_time_robotarm)*1000)
print('gradient STD_robotarm = ', np.std(gradient_time_robotarm)*1000)
print('estimator_time_robotarm = ', np.mean(estimator_time_robotarm)*1000)
print('estimator STD_robotarm = ', np.std(estimator_time_robotarm)*1000)

print('OCIL_time_robotarm = ', np.mean(OCIL_time_robotarm)*1000)
print('OCIL STD_robotarm = ', np.std(OCIL_time_robotarm)*1000)
print('OCIL gradient_time_robotarm = ', np.mean(OCIL_gradient_time_robotarm)*1000)
print('OCIL gradient STD_robotarm = ', np.std(OCIL_gradient_time_robotarm)*1000)

SOCIL_Loss_avg_robotarm = np.mean(Loss_SOCIL_robotarm, axis=0)
SOCIL_Loss_std_robotarm = np.std(Loss_SOCIL_robotarm, axis=0)
SOCIL_Loss_ub_robotarm = SOCIL_Loss_avg_robotarm + 3*SOCIL_Loss_std_robotarm
SOCIL_Loss_lb_robotarm = SOCIL_Loss_avg_robotarm - 3*SOCIL_Loss_std_robotarm
t_SOCIL_robotarm = np.arange(0, len(Loss_SOCIL_robotarm[0]))

for i in range(1,trails):
    results_file = f'examples/robotarm/data/results/SPDP/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 1:
        Loss_SPDP_robotarm = results['loss_trace_barrierOC']
    else:
        # remove bad runs from SPDP
        if np.mean(results['loss_trace_barrierOC'][0]) < 1000:
            Loss_SPDP_robotarm = np.vstack((Loss_SPDP_robotarm, results['loss_trace_barrierOC']))


SPDP_Loss_avg_robotarm = np.mean(Loss_SPDP_robotarm, axis=0)
SPDP_Loss_std_robotarm = np.std(Loss_SPDP_robotarm, axis=0)
SPDP_Loss_ub_robotarm = SPDP_Loss_avg_robotarm + 3*SPDP_Loss_std_robotarm
SPDP_Loss_lb_robotarm = SPDP_Loss_avg_robotarm - 3*SPDP_Loss_std_robotarm
t_SPDP_robotarm = list(range(0, len(Loss_SPDP_robotarm[0])))
t_SPDP_robotarm = [x*horizon_robotarm for x in t_SPDP_robotarm]

EQLQR_trails = 100
for i in range(EQLQR_trails):
    results_file = f'examples/robotarm/data/results/EQLQR/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_EQLQR = results['Loss']
    else:
        Loss_EQLQR = np.vstack((Loss_EQLQR, results['Loss']))
EQLQR_Loss_avg = np.mean(Loss_EQLQR, axis=0)
EQLQR_Loss_std = np.std(Loss_EQLQR, axis=0)
EQLQR_Loss_ub = EQLQR_Loss_avg + 3*EQLQR_Loss_std
EQLQR_Loss_lb = EQLQR_Loss_avg - 3*EQLQR_Loss_std
t_EQLQR = np.arange(0, len(Loss_EQLQR[0]))


fig, axs = plt.subplots(1,2, figsize=(16, 5))

# CartPole plot (left subplot)
axs[0].plot(t_SOCIL[:horizon], SOCIL_Loss_avg[:horizon], 'b-', linewidth=5, label='Safe OCIL, Proposed (Online)')
axs[0].plot(t_SOCIL, SOCIL_Loss_avg, 'b--', linewidth=5, label='Safe OCIL, Proposed (Offline)')
axs[0].plot(t_SPDP, SPDP_Loss_avg, 'r--o', linewidth=5, markersize=12, label='Safe PDP')
axs[0].plot(t_SOCIL[:horizon], SOCIL_Loss_avg[:horizon], 'b-', linewidth=5)
axs[0].plot(t_SOCIL, SOCIL_Loss_avg, 'b--', linewidth=5)
axs[0].fill_between(t_SPDP, SPDP_Loss_lb, SPDP_Loss_ub, color='lightcoral')
axs[0].fill_between(t_SOCIL, SOCIL_Loss_lb, SOCIL_Loss_ub, color='lightskyblue')
axs[0].set_xlabel('# of Data', labelpad=0)
axs[0].set_ylabel('Cartpole Loss', labelpad=0)
axs[0].grid(True, alpha=0.3)
axs[0].set_xlim([0,174])
axs[0].set_ylim([1,500])
axs[0].set_yscale('log')

# RobotArm plot (right subplot)
axs[1].plot(t_SOCIL_robotarm[:horizon_robotarm], SOCIL_Loss_avg_robotarm[:horizon_robotarm], 'b-', linewidth=5, label='Safe OCIL, Proposed (Online)')
axs[1].plot(t_SOCIL_robotarm, SOCIL_Loss_avg_robotarm, 'b--', linewidth=5, label='Safe OCIL, Proposed (Offline)')
axs[1].plot(t_EQLQR[:horizon], EQLQR_Loss_avg[:horizon], 'g-', linewidth=5, label='OCIL w/ Equality-constrained LQR (Online)')
axs[1].plot(t_EQLQR, EQLQR_Loss_avg, 'g--', linewidth=5, label='OCIL w/ Equality-constrained LQR (Offline)')
axs[1].plot(t_SPDP_robotarm, SPDP_Loss_avg_robotarm, 'r--o', linewidth=5, markersize=12, label='Safe PDP')

axs[1].plot(t_SOCIL_robotarm[:horizon_robotarm], SOCIL_Loss_avg_robotarm[:horizon_robotarm], 'b-', linewidth=5)

axs[1].fill_between(t_SPDP_robotarm, SPDP_Loss_lb_robotarm, SPDP_Loss_ub_robotarm, color='lightcoral')
axs[1].fill_between(t_EQLQR, EQLQR_Loss_lb, EQLQR_Loss_ub, color='lightgreen')
axs[1].fill_between(t_SOCIL_robotarm, SOCIL_Loss_lb_robotarm, SOCIL_Loss_ub_robotarm, color='lightskyblue')
axs[1].set_xlabel('# of Data', labelpad=0)
axs[1].set_ylabel('Robot-Arm Loss', labelpad=0)
axs[1].grid(True, alpha=0.3)
axs[1].set_xlim([0,124])
axs[1].set_ylim([0.8,200])
axs[1].set_yscale('log')

# Create unified legend above the figure with two rows
handles, labels = axs[1].get_legend_handles_labels()
# Remove duplicates while preserving order
seen = set()
unique_handles = []
unique_labels = []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)

fig.legend(unique_handles, unique_labels, loc='upper center', ncol=3, fontsize=20, frameon=True, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.80)
plt.show()

# fig, axs = plt.subplots(1,1, figsize=(8, 4))

# # CartPole plot (left subplot)
# axs.plot(t_SOCIL[:horizon], SOCIL_Loss_avg[:horizon], 'b-', linewidth=3)
# axs.plot(t_SOCIL, SOCIL_Loss_avg, 'b--', linewidth=3)
# axs.plot(t_SPDP, SPDP_Loss_avg, 'r--o', linewidth=3, markersize=12)
# axs.fill_between(t_SPDP, SPDP_Loss_lb, SPDP_Loss_ub, color='lightcoral')
# axs.fill_between(t_SOCIL, SOCIL_Loss_lb, SOCIL_Loss_ub, color='lightskyblue')
# axs.set_xlabel('# of Data', labelpad=0)
# axs.set_ylabel('Cartpole Loss', labelpad=0)
# axs.grid(True, alpha=0.3)
# axs.set_xlim([0,174])
# axs.set_ylim([1,500])
# axs.set_yscale('log')
# axs.legend(['Safe OCIL (Online)', 'Safe OCIL (Offline)', 'Safe PDP'], fontsize=18)

# fig, axs = plt.subplots(1,1, figsize=(8, 4))
# # RobotArm plot (right subplot)
# axs.plot(t_SOCIL_robotarm[:horizon_robotarm], SOCIL_Loss_avg_robotarm[:horizon_robotarm], 'b-', linewidth=3)
# axs.plot(t_SOCIL_robotarm, SOCIL_Loss_avg_robotarm, 'b--', linewidth=3)
# axs.plot(t_SPDP_robotarm, SPDP_Loss_avg_robotarm, 'r--o', linewidth=3, markersize=12)
# axs.fill_between(t_SPDP_robotarm, SPDP_Loss_lb_robotarm, SPDP_Loss_ub_robotarm, color='lightcoral')
# axs.fill_between(t_SOCIL_robotarm, SOCIL_Loss_lb_robotarm, SOCIL_Loss_ub_robotarm, color='lightskyblue')
# axs.set_xlabel('# of Data', labelpad=0)
# axs.set_ylabel('Robot-Arm Loss', labelpad=0)
# axs.grid(True, alpha=0.3)
# axs.set_xlim([0,124])
# axs.set_ylim([0.8,200])
# axs.set_yscale('log')
# # axs.legend(['Safe OCIL (Online)', 'Safe OCIL (Offline)', 'Safe PDP'], fontsize=16)

# plt.tight_layout()
# plt.subplots_adjust(hspace=0.1)
# plt.show()

for i in range(trails):
    results_file = f'examples/cartpole/data/results/SOCIL_noise4/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_SOCIL_4 = results['Loss']
    else:
        Loss_SOCIL_4 = np.vstack((Loss_SOCIL_4, results['Loss']))

SOCIL_Loss_4_avg = np.mean(Loss_SOCIL_4, axis=0)
SOCIL_Loss_4_std = np.std(Loss_SOCIL_4, axis=0)
SOCIL_Loss_4_ub = SOCIL_Loss_4_avg + 3*SOCIL_Loss_4_std
SOCIL_Loss_4_lb = SOCIL_Loss_4_avg - 3*SOCIL_Loss_4_std
t_SOCIL_4 = np.arange(0, len(Loss_SOCIL_4[0]))

for i in range(trails):
    results_file = f'examples/cartpole/data/results/SOCIL_noise08/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_SOCIL_08 = results['Loss']
    else:
        Loss_SOCIL_08 = np.vstack((Loss_SOCIL_08, results['Loss']))

SOCIL_Loss_08_avg = np.mean(Loss_SOCIL_08, axis=0)
SOCIL_Loss_08_std = np.std(Loss_SOCIL_08, axis=0)
SOCIL_Loss_08_ub = SOCIL_Loss_08_avg + 3*SOCIL_Loss_08_std
SOCIL_Loss_08_lb = SOCIL_Loss_08_avg - 3*SOCIL_Loss_08_std
t_SOCIL_08 = np.arange(0, len(Loss_SOCIL_08[0]))

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(t_SOCIL, SOCIL_Loss_avg, 'b-', linewidth=3)
plt.plot(t_SOCIL_4, SOCIL_Loss_4_avg, 'g--', linewidth=3)
plt.plot(t_SOCIL_08, SOCIL_Loss_08_avg, '-', color='yellow', linewidth=5)
plt.plot(t_SOCIL, SOCIL_Loss_avg, 'b-', linewidth=3)
plt.plot(t_SOCIL_4, SOCIL_Loss_4_avg, 'g--', linewidth=3)
axs.fill_between(t_SOCIL_08, SOCIL_Loss_08_lb, SOCIL_Loss_08_ub, color='yellow')
axs.fill_between(t_SOCIL_4, SOCIL_Loss_4_lb, SOCIL_Loss_4_ub, color='lightcoral')
axs.fill_between(t_SOCIL, SOCIL_Loss_lb, SOCIL_Loss_ub, color='lightskyblue')


plt.xlabel('# of Data', fontsize=24)
plt.ylabel('Loss', fontsize=24)
# plt.title('Safe OCIL with Different Noise Levels', fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
axs.set_xlim([0,35])
axs.set_ylim([1,1000])
axs.set_yscale('log')
sigmas = [0.0, 0.4, 0.8]
axs.legend([fr'$\sigma={s}$' for s in sigmas], fontsize=24)
plt.tight_layout()
plt.show()


demo_control = SOCIL_results['demo_control']
demo_state = SOCIL_results['demo_state'][:,0]
demo_state_original = SOCIL_results['demo_state_original'][:,0]

SOCIL_control = SOCIL_results['control'][-1]
SOCIL_state = SOCIL_results['state'][-1][:,0]

t_control = np.arange(0, len(demo_control))
t_state = np.arange(0, len(demo_state))
OCIL_control = OCIL_results['control'][-1]
OCIL_state = OCIL_results['state'][-1][:,0]

# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(t_control, demo_control, 'r--', linewidth=3)
# plt.plot(t_control, SOCIL_control, 'b-', linewidth=3)
# plt.plot(t_control, OCIL_control, 'g-', linewidth=3)
# plt.axhline(y=5, color='k', linestyle='--', linewidth=5)
# plt.axhline(y=-5, color='k', linestyle='--', linewidth=5)
# axs.fill_between([-1,35], 5, -5, color='#EFEFEF', alpha=1)
# plt.plot(t_control, demo_control, 'r--', linewidth=3)

# plt.xlabel(f'{r'$t$'}', fontsize=24)
# plt.ylabel(f'{r'$u$'}', fontsize=24)
# # plt.title('CartPole', fontsize=24)
# plt.grid(True, alpha=0.3)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# axs.set_xlim([-1,35])
# plt.tight_layout()
# plt.legend(['Demonstration', 'Safe OCIL', 'OCIL', 'Safety Boundary'], fontsize=24)


# fig, axs = plt.subplots(1, 1, figsize=(10, 8))

# # axs.plot(t_state, demo_state, 'r--', linewidth=3)
# axs.plot(t_state, demo_state_original, 'r--', linewidth=3)
# axs.plot(t_state, SOCIL_state, 'b-', linewidth=3)
# axs.plot(t_state, OCIL_state, 'g-', linewidth=3)
# plt.axhline(y=0.8, color='k', linestyle='--', linewidth=5)
# plt.axhline(y=-0.8, color='k', linestyle='--', linewidth=5)
# axs.fill_between([-1,36], -0.8, 0.8, color='#EFEFEF', alpha=1)
# axs.set_ylabel(f'{r'$p$'}', fontsize=24)
# # axs.set_title('State Trajectories', fontsize=24)
# axs.set_xlabel(f'{r'$t$'}', fontsize=24)
# plt.tight_layout()
# plt.grid(True, alpha=0.3)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# axs.set_xlim([-1,36])
# axs.set_ylim([-2,2])
# axs.legend(['Demonstration', 'Safe OCIL', 'OCIL', 'Safety Boundary'], fontsize=24)

# plt.show()


results_file = f'examples/cartpole/data/results/SOCIL03.mat'
SOCIL03_results = sio.loadmat(results_file)

SOCIL03_control = SOCIL03_results['control'][-1]
SOCIL03_state = SOCIL03_results['state'][-1][:,0]
SOCIL03_demo_original = SOCIL03_results['demo_state_original'][:,0]
SOCIL03_demo = SOCIL03_results['demo_state'][:,0]

results_file = f'examples/cartpole/data/results/SOCIL06.mat'
SOCIL06_results = sio.loadmat(results_file)

SOCIL06_control = SOCIL06_results['control'][-1]
SOCIL06_state = SOCIL06_results['state'][-1][:,0]
t_control06 = np.arange(0, len(SOCIL06_control))
t_state06 = np.arange(0, len(SOCIL06_state))

# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(t_control, demo_control, 'r--', linewidth=3)
# # plt.plot(t_control, SOCIL_control, 'b--', linewidth=3)
# plt.plot(t_control, SOCIL03_control, 'g-', linewidth=3)
# plt.plot(t_control06, SOCIL06_control, 'y--', linewidth=3)
# plt.axhline(y=5, color='k', linestyle='--', linewidth=5)
# plt.axhline(y=-5, color='k', linestyle='--', linewidth=5)
# # plt.plot(t_control, SOCIL_control, 'b--', linewidth=3)
# axs.fill_between([-1,35], 5, -5, color='#EFEFEF', alpha=1)
# plt.plot(t_control, demo_control, 'r--', linewidth=3)

# plt.xlabel(f'{r'$t$'}', fontsize=24)
# plt.ylabel(f'{r'$u$'}', fontsize=24)
# # plt.title('CartPole', fontsize=24)
# plt.grid(True, alpha=0.3)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# axs.set_xlim([-1,35])
# plt.tight_layout()
# plt.legend(['Demonstration', fr'$\sigma=0.4$', fr'$\sigma=0.8$', 'Safety Boundary'], fontsize=24)


# fig, axs = plt.subplots(1, 1, figsize=(10, 8))

# axs.plot(t_state, SOCIL03_demo, 'r--', linewidth=3)
# axs.plot(t_state, SOCIL03_demo_original, 'r-', linewidth=3)
# # axs.plot(t_state, SOCIL_state, 'b--', linewidth=3)
# axs.plot(t_state, SOCIL03_state, 'g-', linewidth=3)
# axs.plot(t_state06, SOCIL06_state, 'y--', linewidth=3)
# plt.axhline(y=0.8, color='k', linestyle='--', linewidth=5)
# plt.axhline(y=-0.8, color='k', linestyle='--', linewidth=5)
# # axs.plot(t_state, SOCIL_state, 'b--', linewidth=3)
# axs.fill_between([-1,36], -0.8, 0.8, color='#EFEFEF', alpha=1)
# axs.set_ylabel(f'{r'$p$'}', fontsize=24)
# # axs.set_title('State Trajectories', fontsize=24)
# axs.set_xlabel(f'{r'$t$'}', fontsize=24)
# plt.tight_layout()
# plt.grid(True, alpha=0.3)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# axs.set_xlim([-1,36])
# axs.set_ylim([-1.5,2])
# axs.legend(['Noisy Measurement', 'Ground Truth', fr'$\sigma=0.4$', fr'$\sigma=0.8$', 'Safety Boundary'], fontsize=24)

# plt.show()


results_file = f'examples/cartpole/data/results/OCIL03.mat'
OCIL03_results = sio.loadmat(results_file)

OCIL03_control = OCIL03_results['control'][-1]
OCIL03_state = OCIL03_results['state'][-1][:,0]


fig, axs = plt.subplots(2, 1, figsize=(8, 8))
# State plot (top subplot)
axs[0].plot(t_state, SOCIL03_demo, 'r--', linewidth=5, label='Noisy Measurement')
axs[0].plot(t_state, SOCIL03_demo_original, 'r-', linewidth=5, label='Ground Truth')
axs[0].plot(t_state, SOCIL03_state, 'b-', linewidth=5, label='Safe OCIL')
axs[0].plot(t_state, OCIL03_state, 'g-', linewidth=5, label='OCIL')
axs[0].axhline(y=0.8, color='k', linestyle='--', linewidth=5)
axs[0].axhline(y=-0.8, color='k', linestyle='--', linewidth=5)
axs[0].fill_between([-1,36], -0.8, 0.8, color='#EFEFEF', alpha=1)
axs[0].set_ylabel(f'{r'$p$'}', labelpad=0)
axs[0].grid(True, alpha=0.3)
axs[0].yaxis.set_major_locator(MultipleLocator(1))
axs[0].set_xlim([-1,36])
axs[0].set_ylim([-2,2])
# axs[0].legend(['Noisy Measurement', 'Ground Truth', 'Safe OCIL', 'OCIL'], fontsize=18)

# Control plot (bottom subplot)
axs[1].plot(t_control, demo_control, 'r--', linewidth=5)
axs[1].plot(t_control, demo_control, 'r-', linewidth=5)
axs[1].plot(t_control, SOCIL03_control, 'b-', linewidth=5)
axs[1].plot(t_control, OCIL03_control, 'g-', linewidth=5)
axs[1].axhline(y=5, color='k', linestyle='--', linewidth=5)
axs[1].axhline(y=-5, color='k', linestyle='--', linewidth=5)
axs[1].fill_between([-1,35], 5, -5, color='#EFEFEF', alpha=1)
axs[1].set_xlabel(f'{r'$t$'}', labelpad=0)
axs[1].set_ylabel(f'{r'$u$'}', labelpad=0)
axs[1].grid(True, alpha=0.3)
axs[1].set_xlim([-1,35])
axs[1].legend(['Noisy Measurement', 'Ground Truth', 'Safe OCIL', 'OCIL'], fontsize=20)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()