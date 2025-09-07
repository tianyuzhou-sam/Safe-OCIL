import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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


print('SOCIL_time = ', np.mean(SOCIL_time)*1000)
print('SOCIL STD = ', np.std(SOCIL_time)*1000)
print('gradient_time = ', np.mean(gradient_time)*1000)
print('gradient STD = ', np.std(gradient_time)*1000)
print('estimator_time = ', np.mean(estimator_time)*1000)
print('estimator STD = ', np.std(estimator_time)*1000)

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


fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(t_SOCIL[:horizon], SOCIL_Loss_avg[:horizon], 'b-', linewidth=3)
plt.plot(t_SOCIL, SOCIL_Loss_avg, 'b--', linewidth=3)
plt.plot(t_SPDP, SPDP_Loss_avg, 'r--o', linewidth=3, markersize=12)
# plt.axvline(x=horizon, color='k', linestyle='--', linewidth=3, alpha=0.5)
axs.fill_between(t_SPDP, SPDP_Loss_lb, SPDP_Loss_ub, color='lightcoral')
axs.fill_between(t_SOCIL, SOCIL_Loss_lb, SOCIL_Loss_ub, color='lightskyblue')
    
plt.xlabel('# of Data', fontsize=24)
plt.ylabel('Loss', fontsize=24)
# plt.title('CartPole', fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
axs.set_xlim([0,174])
axs.set_ylim([1,500])
axs.set_yscale('log')
plt.legend(['Safe OCIL (Online)', 'Safe OCIL (Offline)', 'Safe PDP'], fontsize=24)
plt.tight_layout()
plt.show()

for i in range(trails):
    results_file = f'examples/cartpole/data/results/SOCIL_noise2/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_SOCIL_2 = results['Loss']
    else:
        Loss_SOCIL_2 = np.vstack((Loss_SOCIL_2, results['Loss']))

SOCIL_Loss_2_avg = np.mean(Loss_SOCIL_2, axis=0)
SOCIL_Loss_2_std = np.std(Loss_SOCIL_2, axis=0)
SOCIL_Loss_2_ub = SOCIL_Loss_2_avg + 3*SOCIL_Loss_2_std
SOCIL_Loss_2_lb = SOCIL_Loss_2_avg - 3*SOCIL_Loss_2_std
t_SOCIL_2 = np.arange(0, len(Loss_SOCIL_2[0]))

for i in range(trails):
    results_file = f'examples/cartpole/data/results/SOCIL_noise5/results_{i+1}.mat'
    results = sio.loadmat(results_file)
    if i == 0:
        Loss_SOCIL_5 = results['Loss']
    else:
        Loss_SOCIL_5 = np.vstack((Loss_SOCIL_5, results['Loss']))

SOCIL_Loss_5_avg = np.mean(Loss_SOCIL_5, axis=0)
SOCIL_Loss_5_std = np.std(Loss_SOCIL_5, axis=0)
SOCIL_Loss_5_ub = SOCIL_Loss_5_avg + 3*SOCIL_Loss_5_std
SOCIL_Loss_5_lb = SOCIL_Loss_5_avg - 3*SOCIL_Loss_5_std
t_SOCIL_5 = np.arange(0, len(Loss_SOCIL_5[0]))

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(t_SOCIL, SOCIL_Loss_avg, 'b-', linewidth=4)
plt.plot(t_SOCIL_2, SOCIL_Loss_2_avg, 'r-', linewidth=3)
plt.plot(t_SOCIL_5, SOCIL_Loss_5_avg, 'g-', linewidth=2)
axs.fill_between(t_SOCIL_5, SOCIL_Loss_5_lb, SOCIL_Loss_5_ub, color='lightgreen')
axs.fill_between(t_SOCIL_2, SOCIL_Loss_2_lb, SOCIL_Loss_2_ub, color='lightcoral')
axs.fill_between(t_SOCIL, SOCIL_Loss_lb, SOCIL_Loss_ub, color='lightskyblue')


plt.xlabel('# of Data', fontsize=24)
plt.ylabel('Loss', fontsize=24)
# plt.title('Safe OCIL with Different Noise Levels', fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
axs.set_xlim([0,25])
axs.set_ylim([1,1000])
axs.set_yscale('log')
sigmas = [0, 0.2, 0.5]
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

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(t_control, demo_control, 'r--', linewidth=3)
plt.plot(t_control, SOCIL_control, 'b-', linewidth=3)
plt.plot(t_control, OCIL_control, 'g-', linewidth=3)
plt.axhline(y=5, color='k', linestyle='--', linewidth=5)
plt.axhline(y=-5, color='k', linestyle='--', linewidth=5)
axs.fill_between([-1,35], 5, -5, color='#EFEFEF', alpha=1)
plt.plot(t_control, demo_control, 'r--', linewidth=3)

plt.xlabel(f'{r'$t$'}', fontsize=24)
plt.ylabel(f'{r'$u$'}', fontsize=24)
# plt.title('CartPole', fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
axs.set_xlim([-1,35])
plt.tight_layout()
plt.legend(['Demonstration', 'Safe OCIL', 'OCIL', 'Safety Boundary'], fontsize=24)


fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(t_state, demo_state, 'r--', linewidth=3)
axs.plot(t_state, demo_state_original, 'r-', linewidth=3)
axs.plot(t_state, SOCIL_state, 'b-', linewidth=3)
axs.plot(t_state, OCIL_state, 'g-', linewidth=3)
plt.axhline(y=0.8, color='k', linestyle='--', linewidth=5)
plt.axhline(y=-0.8, color='k', linestyle='--', linewidth=5)
axs.fill_between([-1,36], -0.8, 0.8, color='#EFEFEF', alpha=1)
axs.set_ylabel(f'{r'$p$'}', fontsize=24)
# axs.set_title('State Trajectories', fontsize=24)
axs.set_xlabel(f'{r'$t$'}', fontsize=24)
plt.tight_layout()
plt.grid(True, alpha=0.3)
axs.set_xlim([-1,36])
axs.set_ylim([-2,2])
axs.legend(['Noisy Measurement', 'Ground Truth', 'Safe OCIL', 'OCIL', 'Safety Boundary'], fontsize=24)

plt.show()

    

results_file = f'examples/cartpole/data/results/SOCIL02.mat'
SOCIL02_results = sio.loadmat(results_file)

SOCIL02_control = SOCIL02_results['control'][-1]
SOCIL02_state = SOCIL02_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/SOCIL05.mat'
SOCIL05_results = sio.loadmat(results_file)

SOCIL05_control = SOCIL05_results['control'][-1]
SOCIL05_state = SOCIL05_results['state'][-1][:,0]

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(t_control, demo_control, 'r--', linewidth=3)
plt.plot(t_control, SOCIL_control, 'b--', linewidth=3)
plt.plot(t_control, SOCIL02_control, 'g-', linewidth=3)
plt.plot(t_control, SOCIL05_control, 'y-', linewidth=3)
plt.axhline(y=5, color='k', linestyle='--', linewidth=5)
plt.axhline(y=-5, color='k', linestyle='--', linewidth=5)
plt.plot(t_control, SOCIL02_control, 'g-', linewidth=3)
plt.plot(t_control, SOCIL_control, 'b--', linewidth=3)
axs.fill_between([-1,35], 5, -5, color='#EFEFEF', alpha=1)
plt.plot(t_control, demo_control, 'r--', linewidth=3)

plt.xlabel(f'{r'$t$'}', fontsize=24)
plt.ylabel(f'{r'$u$'}', fontsize=24)
# plt.title('CartPole', fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
axs.set_xlim([-1,35])
plt.tight_layout()
plt.legend(['Demonstration', fr'$\sigma=0$', fr'$\sigma=0.2$', fr'$\sigma=0.5$', 'Safety Boundary'], fontsize=24)


fig, axs = plt.subplots(1, 1, figsize=(10, 8))

axs.plot(t_state, demo_state, 'r--', linewidth=3)
axs.plot(t_state, demo_state_original, 'r-', linewidth=3)
axs.plot(t_state, SOCIL_state, 'b--', linewidth=3)
axs.plot(t_state, SOCIL02_state, 'g-', linewidth=3)
axs.plot(t_state, SOCIL05_state, 'y-', linewidth=3)
plt.axhline(y=0.8, color='k', linestyle='--', linewidth=5)
plt.axhline(y=-0.8, color='k', linestyle='--', linewidth=5)
axs.plot(t_state, SOCIL02_state, 'g-', linewidth=3)
axs.plot(t_state, SOCIL_state, 'b--', linewidth=3)
axs.fill_between([-1,36], -0.8, 0.8, color='#EFEFEF', alpha=1)
axs.set_ylabel(f'{r'$p$'}', fontsize=24)
# axs.set_title('State Trajectories', fontsize=24)
axs.set_xlabel(f'{r'$t$'}', fontsize=24)
plt.tight_layout()
plt.grid(True, alpha=0.3)
axs.set_xlim([-1,36])
axs.set_ylim([-1.5,2])
axs.legend(['Noisy Measurement', 'Ground Truth', fr'$\sigma=0$', fr'$\sigma=0.2$', fr'$\sigma=0.5$', 'Safety Boundary'], fontsize=24)

plt.show()