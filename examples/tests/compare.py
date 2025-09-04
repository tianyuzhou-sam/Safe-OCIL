import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import SafeOCIL
import JinEnv
import generateTraj
import ocSolver
import PDP
import SafePDP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mc = 0.5
mp = 0.5
l = 1
wx = 0.1
wq = 1
wdx = 0.1
wdq = 0.1
max_u = 10

dynsys = JinEnv.CartPole()
dynsys.initDyn()
dynsys.initCost(wu=0.1)
dynsys.initConstraints2()

dt = 0.1
horizon = 35
init_state = [0,0,0,0]

true_parameter = [mc, mp, l, wx, wq, wdx, wdq, max_u]

# Use ocSolver to solve for trajectory and auxiliary trajectory
coc = SafePDP.COCsys()
# pass the system to coc
coc.setAuxvarVariable(vertcat(dynsys.dyn_auxvar, dynsys.cost_auxvar, dynsys.constraint_auxvar))
print("Auxiliary variables:", coc.auxvar)
coc.setStateVariable(dynsys.X)
coc.setControlVariable(dynsys.U)
# Set discretized dynamics: self.dyn = env.X + dt * env.f
dyn = dynsys.X + dt * dynsys.f
coc.setDyn(dyn)
# pass cost to coc
coc.setPathCost(dynsys.path_cost)
coc.setFinalCost(dynsys.final_cost)
# pass constraints to coc
coc.setPathInequCstr(dynsys.path_inequ)
# differentiating CPMP
traj_COC = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=true_parameter)

coc.diffCPMP()

# auxsys_COC = coc.getAuxSys(opt_sol=traj_COC, threshold=1e-5)
# clqr = ocSolver.EQCLQR()
# clqr.auxsys2Eqctlqr(auxsys=auxsys_COC)
# aux_sol_COC = clqr.eqctlqrSolver(threshold=1e-5) 

bad_solution_threshold = 100000

print("Trajectory solved successfully!")
print("State trajectory shape:", traj_COC['state_traj_opt'].shape)
print("Control trajectory shape:", traj_COC['control_traj_opt'].shape)
print("Auxiliary system solved successfully!")

# Create separate state and control trajectories for COC
coc_state_traj = traj_COC['state_traj_opt']
coc_control_traj = traj_COC['control_traj_opt']

# Create barrier system using the SafeOCIL pattern
coc_barrier = ocSolver.OCSys()
coc_barrier.setAuxvarVariable(vertcat(dynsys.dyn_auxvar, dynsys.cost_auxvar, dynsys.constraint_auxvar))
coc_barrier.setStateVariable(dynsys.X)
coc_barrier.setControlVariable(dynsys.U)
# Set discretized dynamics: self.dyn = env.X + dt * env.f
dyn = dynsys.X + dt * dynsys.f
coc_barrier.setDyn(dyn)
coc_barrier.setPathCost(dynsys.path_cost)
coc_barrier.setFinalCost(dynsys.final_cost)
coc_barrier.setPathInequCstr(dynsys.path_inequ)
coc_barrier.diffCPMP()

# Convert to barrier system using the SafeOCIL pattern
coc_barrier.convert2BarrierOC(alpha=4e-2, beta=1e-2)

# Solve barrier system using the SafeOCIL pattern
traj_barrier = coc_barrier.solveBarrierOC(horizon=horizon, ini_state=init_state, auxvar_value=true_parameter)
# traj_barrier = coc_barrier.solveBarrierOCRef(ini_state=init_state, horizon=horizon, auxvar_value = true_parameter, ref=ref)

# Extract state and control trajectories for barrier system
barrier_state_traj = traj_barrier['state_traj_opt']
barrier_control_traj = traj_barrier['control_traj_opt']

# Compute auxiliary system for barrier using the SafeOCIL pattern
aux_sol_COC = coc_barrier.auxSysBarrierOC(opt_sol=traj_barrier)

# Compute auxiliary solution difference norm using separate state and control norms
# Extract trajectory arrays from the dictionaries for comparison
# Convert to numpy arrays first
aux_coc_state = np.array(aux_sol_COC['state_traj_opt'])
aux_coc_control = np.array(aux_sol_COC['control_traj_opt'])

# Define alpha and beta ranges for barrier system (reverse order: largest to smallest)
alpha_range = 4*np.logspace(-2, -1, 5)
beta_range = np.logspace(-2, -1, 5)

# Initialize arrays to store differences
traj_diff_norms = np.zeros((len(alpha_range), len(beta_range)))
aux_diff_norms = np.zeros((len(alpha_range), len(beta_range)))

# refsys = JinEnv.CartPole()
# refsys.initDyn(mc=0.5, mp=0.5, l=1)
# refsys.initCost(wx=1, wq=6, wdx=1, wdq=1, wu = 0.1)
# refoc = ocSolver.OCSys()
# # pass the system to coc
# refoc.setAuxvarVariable(vertcat(refsys.dyn_auxvar, refsys.cost_auxvar))
# refoc.setStateVariable(refsys.X)
# refoc.setControlVariable(refsys.U)
# # Set discretized dynamics: self.dyn = env.X + dt * env.f
# refdyn = refsys.X + dt * refsys.f
# refoc.setDyn(refdyn)
# # pass cost to coc
# refoc.setPathCost(refsys.path_cost)
# refoc.setFinalCost(refsys.final_cost)
# # differentiating CPMP
# ref = refoc.ocSolver(horizon=horizon, ini_state=init_state, auxvar_value=[])
# ref = None

for i, alpha in enumerate(alpha_range):
    for j, beta in enumerate(beta_range):
        print(f"Processing [{i+1},{j+1}/100]: alpha={alpha:.2e}, beta={beta:.2e}")
        
        try:
            # Create barrier system using the SafeOCIL pattern
            coc_barrier = ocSolver.OCSys()
            coc_barrier.setAuxvarVariable(vertcat(dynsys.dyn_auxvar, dynsys.cost_auxvar, dynsys.constraint_auxvar))
            coc_barrier.setStateVariable(dynsys.X)
            coc_barrier.setControlVariable(dynsys.U)
            # Set discretized dynamics: self.dyn = env.X + dt * env.f
            dyn = dynsys.X + dt * dynsys.f
            coc_barrier.setDyn(dyn)
            coc_barrier.setPathCost(dynsys.path_cost)
            coc_barrier.setFinalCost(dynsys.final_cost)
            coc_barrier.setPathInequCstr(dynsys.path_inequ)
            coc_barrier.diffCPMP()
            
            # Convert to barrier system using the SafeOCIL pattern
            coc_barrier.convert2BarrierOC(alpha=alpha, beta=beta)
            
            # Solve barrier system using the SafeOCIL pattern
            traj_barrier = coc_barrier.solveBarrierOC(horizon=horizon, ini_state=init_state, auxvar_value=true_parameter)
            # traj_barrier = coc_barrier.solveBarrierOCRef(ini_state=init_state, horizon=horizon, auxvar_value = true_parameter, ref=ref)
            
            # Extract state and control trajectories for barrier system
            barrier_state_traj = traj_barrier['state_traj_opt']
            barrier_control_traj = traj_barrier['control_traj_opt']
            
            # Compute trajectory difference norm using separate state and control norms
            traj_diff_norms[i, j] = np.linalg.norm(coc_state_traj.flatten() - barrier_state_traj.flatten(), 2) ** 2 + np.linalg.norm(
                coc_control_traj.flatten() - barrier_control_traj.flatten(), 2) ** 2
            
            # Compute auxiliary system for barrier using the SafeOCIL pattern
            aux_sol_barrier = coc_barrier.auxSysBarrierOC(opt_sol=traj_barrier)
            
            # Compute auxiliary solution difference norm using separate state and control norms
            # Extract trajectory arrays from the dictionaries for comparison
            # Convert to numpy arrays first
            aux_barrier_state = np.array(aux_sol_barrier['state_traj_opt'])
            aux_barrier_control = np.array(aux_sol_barrier['control_traj_opt'])
            
            aux_diff_norms[i, j] = np.linalg.norm(aux_coc_state.flatten() - aux_barrier_state.flatten(), 2) ** 2 + np.linalg.norm(aux_coc_control.flatten() - aux_barrier_control.flatten(), 2) ** 2

            # if traj_diff_norms[i, j] > bad_solution_threshold:
            #     traj_barrier = coc_barrier.solveBarrierOCRef(ini_state=init_state, horizon=horizon, auxvar_value = [], ref=ref)
            #     # Extract state and control trajectories for barrier system
            #     barrier_state_traj = traj_barrier['state_traj_opt']
            #     barrier_control_traj = traj_barrier['control_traj_opt']
                
            #     # Compute trajectory difference norm using separate state and control norms
            #     traj_diff_norms[i, j] = np.linalg.norm(coc_state_traj.flatten() - barrier_state_traj.flatten(), 2) ** 2 + np.linalg.norm(coc_control_traj.flatten() - barrier_control_traj.flatten(), 2) ** 2
                
            #     # Compute auxiliary system for barrier using the SafeOCIL pattern
            #     aux_sol_barrier = coc_barrier.auxSysBarrierOC(opt_sol=traj_barrier)

            #     # Compute auxiliary solution difference norm using separate state and control norms
            #     # Extract trajectory arrays from the dictionaries for comparison
            #     # Convert to numpy arrays first
            #     aux_barrier_state = np.array(aux_sol_barrier['state_traj_opt'])
            #     aux_barrier_control = np.array(aux_sol_barrier['control_traj_opt'])
                
            #     aux_diff_norms[i, j] = np.linalg.norm(aux_coc_state.flatten() - aux_barrier_state.flatten(), 2) ** 2 + np.linalg.norm(aux_coc_control.flatten() - aux_barrier_control.flatten(), 2) ** 2

            # else:
            #     ref = traj_barrier
            
            # Print results for this alpha/beta combination
            print(f"  Trajectory error: {traj_diff_norms[i, j]:.6f}")
            print(f"  Auxiliary trajectory error: {aux_diff_norms[i, j]:.6f}")

            # plt.plot(coc_control_traj)
            # plt.plot(barrier_control_traj)
            # plt.show()
                        
            
        except Exception as e:
            print(f"Error for alpha={alpha:.2e}, beta={beta:.2e}: {e}")
            traj_diff_norms[i, j] = np.nan
            aux_diff_norms[i, j] = np.nan

# Create line plots
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Line plot for trajectory differences
for j, beta in enumerate(beta_range):
    ax1.plot(alpha_range, traj_diff_norms[:, j], 'o-', linewidth=2, markersize=6, 
             label=f'β = {beta:.1e}')

ax1.set_xlabel('α')
ax1.set_ylabel('Trajectory Difference Norm')
ax1.set_title('Trajectory Difference Norms vs Alpha')
ax1.set_xscale('log')
ax1.set_yscale('log')  # Symmetric log scale that handles zero and negative values
ax1.invert_xaxis()  # Reverse alpha axis: large to small from left to right
ax1.grid(True, alpha=0.3)
ax1.legend()
plt.tight_layout()
plt.show()

# Create second plot for auxiliary solution differences
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Line plot for auxiliary solution differences
for j, beta in enumerate(beta_range):
    ax2.plot(alpha_range, aux_diff_norms[:, j], 's-', linewidth=2, markersize=6, 
             label=f'β = {beta:.1e}')

ax2.set_xlabel('α')
ax2.set_ylabel('Auxiliary Solution Difference Norm')
ax2.set_title('Auxiliary Solution Difference Norms vs Alpha')
ax2.set_xscale('log')
ax2.set_yscale('log')  # Symmetric log scale that handles zero and negative values
ax2.invert_xaxis()  # Reverse alpha axis: large to small from left to right
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Trajectory difference norms - Min: {np.nanmin(traj_diff_norms):.6f}, Max: {np.nanmax(traj_diff_norms):.6f}")
print(f"Auxiliary solution difference norms - Min: {np.nanmin(aux_diff_norms):.6f}, Max: {np.nanmax(aux_diff_norms):.6f}")

# Find alpha and beta values that give minimum differences
if not np.all(np.isnan(traj_diff_norms)):
    min_traj_idx = np.unravel_index(np.nanargmin(traj_diff_norms), traj_diff_norms.shape)
    print(f"Minimum trajectory difference at alpha={alpha_range[min_traj_idx[0]]:.2e}, beta={beta_range[min_traj_idx[1]]:.2e}")

if not np.all(np.isnan(aux_diff_norms)):
    min_aux_idx = np.unravel_index(np.nanargmin(aux_diff_norms), aux_diff_norms.shape)
    print(f"Minimum auxiliary difference at alpha={alpha_range[min_aux_idx[0]]:.2e}, beta={beta_range[min_aux_idx[1]]:.2e}")
