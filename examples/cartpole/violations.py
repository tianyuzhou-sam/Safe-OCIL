import numpy as np
import scipy.io as sio

x_max = 0.8
u_max = 5.0

SOCIL_results_file = 'examples/cartpole/data/results/SOCIL.mat'
SOCIL_results = sio.loadmat(SOCIL_results_file)
OCIL_results_file = 'examples/cartpole/data/results/OCIL.mat'
OCIL_results = sio.loadmat(OCIL_results_file)

demo_control = SOCIL_results['demo_control']
demo_state = SOCIL_results['demo_state'][:,0]
demo_state_original = SOCIL_results['demo_state_original'][:,0]

SOCIL_control = SOCIL_results['control'][-1]
SOCIL_state = SOCIL_results['state'][-1][:,0]

OCIL_control = OCIL_results['control'][-1]
OCIL_state = OCIL_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/SOCIL03.mat'
SOCIL03_results = sio.loadmat(results_file)
SOCIL03_control = SOCIL03_results['control'][-1]
SOCIL03_state = SOCIL03_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/SOCIL06.mat'
SOCIL06_results = sio.loadmat(results_file)
SOCIL06_control = SOCIL06_results['control'][-1]
SOCIL06_state = SOCIL06_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/OCIL03.mat'
OCIL03_results = sio.loadmat(results_file)
OCIL03_control = OCIL03_results['control'][-1]
OCIL03_state = OCIL03_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/OCIL06.mat'
OCIL06_results = sio.loadmat(results_file)
OCIL06_control = OCIL06_results['control'][-1]
OCIL06_state = OCIL06_results['state'][-1][:,0]

horizon = len(demo_control)

SOCIL_control_violations = 0
SOCIL_control_max_violation = 0
SOCIL_state_violations = 0
SOCIL_state_max_violation = 0
for i in range(horizon):
    if abs(SOCIL_control[i]) > u_max:
        OCIL_control_violations += 1
        if abs(SOCIL_control[i]) - u_max > SOCIL_control_max_violation:
            SOCIL_control_max_violation = abs(SOCIL_control[i]) - u_max
    if abs(SOCIL_state[i+1]) > x_max:
        SOCIL_state_violations += 1
        if abs(SOCIL_state[i+1]) - x_max > SOCIL_state_max_violation:
            SOCIL_state_max_violation = abs(SOCIL_state[i+1]) - x_max

print('SOCIL_control_violations = ', SOCIL_control_violations/horizon)
print('SOCIL_control_max_violation = ', SOCIL_control_max_violation/u_max)
print('SOCIL_state_violations = ', SOCIL_state_violations/horizon)
print('SOCIL_state_max_violation = ', SOCIL_state_max_violation/x_max)

OCIL_control_violations = 0
OCIL_control_max_violation = 0
OCIL_state_violations = 0
OCIL_state_max_violation = 0
for i in range(horizon):
    if abs(OCIL_control[i]) > u_max:
        OCIL_control_violations += 1
        if abs(OCIL_control[i]) - u_max > OCIL_control_max_violation:
            OCIL_control_max_violation = abs(OCIL_control[i]) - u_max
    if abs(OCIL_state[i+1]) > x_max:
        OCIL_state_violations += 1
        if abs(OCIL_state[i+1]) - x_max > OCIL_state_max_violation:
            OCIL_state_max_violation = abs(OCIL_state[i+1]) - x_max

print('OCIL_control_violations = ', OCIL_control_violations/horizon)
print('OCIL_control_max_violation = ', OCIL_control_max_violation/u_max)
print('OCIL_state_violations = ', OCIL_state_violations/horizon)
print('OCIL_state_max_violation = ', OCIL_state_max_violation/x_max)

SOCIL03_control_violations = 0
SOCIL03_control_max_violation = 0
SOCIL03_state_violations = 0
SOCIL03_state_max_violation = 0
for i in range(horizon):
    if abs(SOCIL03_control[i]) > u_max:
        SOCIL03_control_violations += 1
        if abs(SOCIL03_control[i]) - u_max > SOCIL03_control_max_violation:
            SOCIL03_control_max_violation = abs(SOCIL03_control[i]) - u_max
    if abs(SOCIL03_state[i+1]) > x_max:
        SOCIL03_state_violations += 1
        if abs(SOCIL03_state[i+1]) - x_max > SOCIL03_state_max_violation:
            SOCIL03_state_max_violation = abs(SOCIL03_state[i+1]) - x_max

print('SOCIL03_control_violations = ', SOCIL03_control_violations/horizon)
print('SOCIL03_control_max_violation = ', SOCIL03_control_max_violation/u_max)
print('SOCIL03_state_violations = ', SOCIL03_state_violations/horizon)
print('SOCIL03_state_max_violation = ', SOCIL03_state_max_violation/x_max)

OCIL03_control_violations = 0
OCIL03_control_max_violation = 0
OCIL03_state_violations = 0
OCIL03_state_max_violation = 0
for i in range(horizon):
    if abs(OCIL03_control[i]) > u_max:
        OCIL03_control_violations += 1
        if abs(OCIL03_control[i]) - u_max > OCIL03_control_max_violation:
            OCIL03_control_max_violation = abs(OCIL03_control[i]) - u_max
    if abs(OCIL03_state[i+1]) > x_max:
        OCIL03_state_violations += 1
        if abs(OCIL03_state[i+1]) - x_max > OCIL03_state_max_violation:
            OCIL03_state_max_violation = abs(OCIL03_state[i+1]) - x_max

print('OCIL03_control_violations = ', OCIL03_control_violations/horizon)
print('OCIL03_control_max_violation = ', OCIL03_control_max_violation/u_max)
print('OCIL03_state_violations = ', OCIL03_state_violations/horizon)
print('OCIL03_state_max_violation = ', OCIL03_state_max_violation/x_max)

SOCIL06_control_violations = 0
SOCIL06_control_max_violation = 0
SOCIL06_state_violations = 0
SOCIL06_state_max_violation = 0
for i in range(horizon):
    if abs(SOCIL06_control[i]) > u_max:
        SOCIL06_control_violations += 1
        if abs(SOCIL06_control[i]) - u_max > SOCIL06_control_max_violation:
            SOCIL06_control_max_violation = abs(SOCIL06_control[i]) - u_max
    if abs(SOCIL06_state[i+1]) > x_max:
        SOCIL06_state_violations += 1
        if abs(SOCIL06_state[i+1]) - x_max > SOCIL06_state_max_violation:
            SOCIL06_state_max_violation = abs(SOCIL06_state[i+1]) - x_max

print('SOCIL06_control_violations = ', SOCIL06_control_violations/horizon)
print('SOCIL06_control_max_violation = ', SOCIL06_control_max_violation/u_max)
print('SOCIL06_state_violations = ', SOCIL06_state_violations/horizon)
print('SOCIL06_state_max_violation = ', SOCIL06_state_max_violation/x_max)

OCIL06_control_violations = 0
OCIL06_control_max_violation = 0
OCIL06_state_violations = 0
OCIL06_state_max_violation = 0
for i in range(horizon):
    if abs(OCIL06_control[i]) > u_max:
        OCIL06_control_violations += 1
        if abs(OCIL06_control[i]) - u_max > OCIL06_control_max_violation:
            OCIL06_control_max_violation = abs(OCIL06_control[i]) - u_max
    if abs(OCIL06_state[i+1]) > x_max:
        OCIL06_state_violations += 1
        if abs(OCIL06_state[i+1]) - x_max > OCIL06_state_max_violation:
            OCIL06_state_max_violation = abs(OCIL06_state[i+1]) - x_max

print('OCIL06_control_violations = ', OCIL06_control_violations/horizon)
print('OCIL06_control_max_violation = ', OCIL06_control_max_violation/u_max)
print('OCIL06_state_violations = ', OCIL06_state_violations/horizon)
print('OCIL06_state_max_violation = ', OCIL06_state_max_violation/x_max)