import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load the results file
trails = 100
for i in range(trails):
    results_file = f'examples/robotarm/data/results/SOCIL/results_{i+1}.mat'
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


print('SOCIL_time = ', np.mean(SOCIL_time))
print('estimator_time = ', np.mean(estimator_time))
print('gradient_time = ', np.mean(gradient_time))

SOCIL_Loss_avg = np.mean(Loss_SOCIL, axis=0)
SOCIL_Loss_std = np.std(Loss_SOCIL, axis=0)
SOCIL_Loss_ub = SOCIL_Loss_avg + 3*SOCIL_Loss_std
SOCIL_Loss_lb = SOCIL_Loss_avg - 3*SOCIL_Loss_std
t_SOCIL = np.arange(0, len(Loss_SOCIL[0]))

for i in range(1,trails):
    results_file = f'examples/robotarm/data/results/SPDP/results_{i+1}.mat'
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



fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(t_SOCIL, SOCIL_Loss_avg, 'b-', linewidth=2)
plt.plot(t_SPDP, SPDP_Loss_avg, 'r-o', linewidth=2)
axs.fill_between(t_SPDP, SPDP_Loss_lb, SPDP_Loss_ub, color='lightcoral')
axs.fill_between(t_SOCIL, SOCIL_Loss_lb, SOCIL_Loss_ub, color='lightskyblue')
    
plt.xlabel('# of Data', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('RobotArm', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
axs.set_xlim([0,124])
axs.set_ylim([0.8,200])
axs.set_yscale('log')
plt.tight_layout()
plt.show()


# demo_control = results['demo_control']
# control_unlearned = results['control'][0]
# control_learned = results['control'][-1]
# x = np.arange(0, len(demo_control))

# # OCIL_control = OCIL_results['control'][-1]

# fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(x, demo_control, 'r--', linewidth=3)
# plt.plot(x, control_learned, 'b-', linewidth=3)
# # plt.plot(x, OCIL_control, 'g--', linewidth=3)
# plt.axhline(y=10, color='k', linestyle='-', linewidth=3)
# plt.axhline(y=-10, color='k', linestyle='-', linewidth=3)
# plt.plot(x, demo_control, 'r--', linewidth=3)
    
# plt.xlabel('# of Data', fontsize=14)
# plt.ylabel('Control', fontsize=14)
# plt.title('CartPole', fontsize=16)
# plt.grid(True, alpha=0.3)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
# plt.legend(['Demo Control', 'Learned Control (Safe OCIL)', 'Learned Control (OCIL)', 'Safety Boundary'], fontsize=12)


# demo_state = results['demo_state']
# demo_state_original = results['demo_state_original']
# state_learned = results['state'][-1]
# x = np.arange(0, len(demo_state))

# state_names = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
# fig, axs = plt.subplots(len(demo_state[0]), 1, figsize=(10, 8))
# for i in range(len(demo_state[0])):
#     axs[i].plot(x, demo_state_original[:,i], 'r--', linewidth=2)
#     axs[i].plot(x, demo_state[:,i], 'g--', linewidth=2)
#     axs[i].plot(x, state_learned[:,i], 'b-', linewidth=2)
#     axs[i].plot(x, demo_state_original[:,i], 'r--', linewidth=2)
#     axs[i].set_ylabel(f'{state_names[i]}', fontsize=14)
# axs[0].set_title('State Trajectories', fontsize=16)
# axs[-1].set_xlabel('# of Data', fontsize=14)
# plt.tight_layout()
# axs[0].legend(['Observed Trajectory', 'Ground Truth', 'Learned Trajectory'], fontsize=12)

# plt.show()

    

