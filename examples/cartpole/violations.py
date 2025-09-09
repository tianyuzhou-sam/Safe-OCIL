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

results_file = f'examples/cartpole/data/results/SOCIL04.mat'
SOCIL04_results = sio.loadmat(results_file)
SOCIL04_control = SOCIL04_results['control'][-1]
SOCIL04_state = SOCIL04_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/SOCIL08.mat'
SOCIL08_results = sio.loadmat(results_file)
SOCIL08_control = SOCIL08_results['control'][-1]
SOCIL08_state = SOCIL08_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/OCIL04.mat'
OCIL04_results = sio.loadmat(results_file)
OCIL04_control = OCIL04_results['control'][-1]
OCIL04_state = OCIL04_results['state'][-1][:,0]

results_file = f'examples/cartpole/data/results/OCIL08.mat'
OCIL08_results = sio.loadmat(results_file)
OCIL08_control = OCIL08_results['control'][-1]
OCIL08_state = OCIL08_results['state'][-1][:,0]

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

SOCIL04_control_violations = 0
SOCIL04_control_max_violation = 0
SOCIL04_state_violations = 0
SOCIL04_state_max_violation = 0
for i in range(horizon):
    if abs(SOCIL04_control[i]) > u_max:
        SOCIL04_control_violations += 1
        if abs(SOCIL04_control[i]) - u_max > SOCIL04_control_max_violation:
            SOCIL04_control_max_violation = abs(SOCIL04_control[i]) - u_max
    if abs(SOCIL04_state[i+1]) > x_max:
        SOCIL04_state_violations += 1
        if abs(SOCIL04_state[i+1]) - x_max > SOCIL04_state_max_violation:
            SOCIL04_state_max_violation = abs(SOCIL04_state[i+1]) - x_max

print('SOCIL04_control_violations = ', SOCIL04_control_violations/horizon)
print('SOCIL04_control_max_violation = ', SOCIL04_control_max_violation/u_max)
print('SOCIL04_state_violations = ', SOCIL04_state_violations/horizon)
print('SOCIL04_state_max_violation = ', SOCIL04_state_max_violation/x_max)

OCIL04_control_violations = 0
OCIL04_control_max_violation = 0
OCIL04_state_violations = 0
OCIL04_state_max_violation = 0
for i in range(horizon):
    if abs(OCIL04_control[i]) > u_max:
        OCIL04_control_violations += 1
        if abs(OCIL04_control[i]) - u_max > OCIL04_control_max_violation:
            OCIL04_control_max_violation = abs(OCIL04_control[i]) - u_max
    if abs(OCIL04_state[i+1]) > x_max:
        OCIL04_state_violations += 1
        if abs(OCIL04_state[i+1]) - x_max > OCIL04_state_max_violation:
            OCIL04_state_max_violation = abs(OCIL04_state[i+1]) - x_max

print('OCIL04_control_violations = ', OCIL04_control_violations/horizon)
print('OCIL04_control_max_violation = ', OCIL04_control_max_violation/u_max)
print('OCIL04_state_violations = ', OCIL04_state_violations/horizon)
print('OCIL04_state_max_violation = ', OCIL04_state_max_violation/x_max)

SOCIL08_control_violations = 0
SOCIL08_control_max_violation = 0
SOCIL08_state_violations = 0
SOCIL08_state_max_violation = 0
for i in range(horizon):
    if abs(SOCIL08_control[i]) > u_max:
        SOCIL08_control_violations += 1
        if abs(SOCIL08_control[i]) - u_max > SOCIL08_control_max_violation:
            SOCIL08_control_max_violation = abs(SOCIL08_control[i]) - u_max
    if abs(SOCIL08_state[i+1]) > x_max:
        SOCIL08_state_violations += 1
        if abs(SOCIL08_state[i+1]) - x_max > SOCIL08_state_max_violation:
            SOCIL08_state_max_violation = abs(SOCIL08_state[i+1]) - x_max

print('SOCIL08_control_violations = ', SOCIL08_control_violations/horizon)
print('SOCIL08_control_max_violation = ', SOCIL08_control_max_violation/u_max)
print('SOCIL08_state_violations = ', SOCIL08_state_violations/horizon)
print('SOCIL08_state_max_violation = ', SOCIL08_state_max_violation/x_max)

OCIL08_control_violations = 0
OCIL08_control_max_violation = 0
OCIL08_state_violations = 0
OCIL08_state_max_violation = 0
for i in range(horizon):
    if abs(OCIL08_control[i]) > u_max:
        OCIL08_control_violations += 1
        if abs(OCIL08_control[i]) - u_max > OCIL08_control_max_violation:
            OCIL08_control_max_violation = abs(OCIL08_control[i]) - u_max
    if abs(OCIL08_state[i+1]) > x_max:
        OCIL08_state_violations += 1
        if abs(OCIL08_state[i+1]) - x_max > OCIL08_state_max_violation:
            OCIL08_state_max_violation = abs(OCIL08_state[i+1]) - x_max

print('OCIL08_control_violations = ', OCIL08_control_violations/horizon)
print('OCIL08_control_max_violation = ', OCIL08_control_max_violation/u_max)
print('OCIL08_state_violations = ', OCIL08_state_violations/horizon)
print('OCIL08_state_max_violation = ', OCIL08_state_max_violation/x_max)