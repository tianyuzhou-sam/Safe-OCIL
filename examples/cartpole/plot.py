import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load the results file
results_file = 'examples/cartpole/data/results/results.mat'
results = sio.loadmat(results_file)

# OCIL_results_file = 'examples/cartpole/data/results/results_OCIL.mat'
# OCIL_results = sio.loadmat(OCIL_results_file)


x = np.arange(0, len(results['Loss'][0]))
Loss = results['Loss'][0]

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(x, Loss, 'b-', linewidth=2)
    
plt.xlabel('# of Data', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('CartPole', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()


demo_control = results['demo_control']
control_unlearned = results['control'][0]
control_learned = results['control'][-1]
x = np.arange(0, len(demo_control))

# OCIL_control = OCIL_results['control'][-1]

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
plt.plot(x, demo_control, 'r--', linewidth=3)
plt.plot(x, control_learned, 'b-', linewidth=3)
# plt.plot(x, OCIL_control, 'g--', linewidth=3)
plt.axhline(y=10, color='k', linestyle='-', linewidth=3)
plt.axhline(y=-10, color='k', linestyle='-', linewidth=3)
plt.plot(x, demo_control, 'r--', linewidth=3)
    
plt.xlabel('# of Data', fontsize=14)
plt.ylabel('Control', fontsize=14)
plt.title('CartPole', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend(['Demo Control', 'Learned Control (Safe OCIL)', 'Learned Control (OCIL)', 'Safety Boundary'], fontsize=12)


demo_state = results['demo_state']
demo_state_original = results['demo_state_original']
state_learned = results['state'][-1]
x = np.arange(0, len(demo_state))

state_names = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
fig, axs = plt.subplots(len(demo_state[0]), 1, figsize=(10, 8))
for i in range(len(demo_state[0])):
    axs[i].plot(x, demo_state_original[:,i], 'r--', linewidth=2)
    axs[i].plot(x, demo_state[:,i], 'g--', linewidth=2)
    axs[i].plot(x, state_learned[:,i], 'b-', linewidth=2)
    axs[i].plot(x, demo_state_original[:,i], 'r--', linewidth=2)
    axs[i].set_ylabel(f'{state_names[i]}', fontsize=14)
axs[0].set_title('State Trajectories', fontsize=16)
axs[-1].set_xlabel('# of Data', fontsize=14)
plt.tight_layout()
axs[0].legend(['Observed Trajectory', 'Ground Truth', 'Learned Trajectory'], fontsize=12)

plt.show()

    

