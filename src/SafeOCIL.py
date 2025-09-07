import numpy as np
from casadi import *
import scipy.io as sio
from scipy.special import softmax
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os, shutil
import sys
import time
import copy
sys.path.append(os.getcwd() + '/src/')
import JinEnv
from EKF import EKF
from Loss_function import Loss
import ocSolver
import PDP


class ImitationLearning:
    def __init__(self, project="", mode="", dynsys=None, noise=None, alpha=1e-2, beta=1e-2, dir="", demoFile="", saveFlag=False):


        self.dir = dir
        self.saveFlag = saveFlag
        self.plotTrajFlag = False
        if saveFlag:
            if not os.path.exists(self.dir+"results/"):
                os.mkdir(self.dir+"results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.mode = mode
        self.dynsys = dynsys
        self.num_dyn_auxvar = dynsys.dyn_auxvar.shape[0]
        self.num_cost_auxvar = dynsys.cost_auxvar.shape[0]
        self.num_constraint_auxvar = dynsys.constraint_auxvar.shape[0]
        self.num_auxvar = self.num_dyn_auxvar + self.num_cost_auxvar + self.num_constraint_auxvar
        
        # ------------------------------ load demos data ------------------------------
        data = sio.loadmat(dir+demoFile)
        self.trajectories = data['trajectories']
        self.dt = data['dt']
        if mode == "Objective":
            self.true_theta = data['true_parameter'].flatten()
            self.true_theta = self.true_theta[len(self.true_theta)-self.num_cost_auxvar:]
        elif mode == "Dynamic":
            self.true_theta = data['true_parameter'].flatten()
            self.true_theta = self.true_theta[:self.num_dyn_auxvar]
        else:
            self.true_theta = data['true_parameter'].flatten()

        print(self.true_theta)
        # self.true_theta = np.hstack((self.true_theta, [10]))

        # ------------------------------ initialize Classes ------------------------------
        self.sysoc = ocSolver.OCSys()
        self.sysoc.setAuxvarVariable(vertcat(self.dynsys.dyn_auxvar, self.dynsys.cost_auxvar, self.dynsys.constraint_auxvar))
        self.sysoc.setControlVariable(self.dynsys.U)
        self.sysoc.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.sysoc.setDyn(self.dyn)
        self.sysoc.setPathCost(self.dynsys.path_cost)
        self.sysoc.setFinalCost(self.dynsys.final_cost)
        self.sysoc.setPathInequCstr(self.dynsys.path_inequ)
        self.sysoc.diffCPMP()

        self.clqr = ocSolver.EQCLQR()
        self.beta = beta
        self.alpha = alpha
        self.sysoc.convert2BarrierOC(alpha=alpha, beta=beta)

        # ------------------------------ initialize task constraint functions ------------------------------
        if hasattr(self.dynsys, 'task_const') and self.dynsys.task_const is not None:
            self.task_const_fn = Function("task_const_fn", [self.dynsys.X, self.dynsys.U], [self.dynsys.task_const])
            self.task_const_num = self.dynsys.task_const.shape[0]
        else:
            self.task_const_fn = None
            self.task_const_num = 0

        self.theta = self.true_theta

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj_original = self.trajectories[0, 0]['state_traj_opt']
        self.demo_control_traj = self.trajectories[0, 0]['control_traj_opt']
        self.demo_ini_state = self.demo_state_traj_original[0, :]
        self.demo_horizon = self.demo_control_traj.shape[0]

        self.demo_state_traj = self.demo_state_traj_original + np.random.normal(0, noise, (self.demo_horizon+1,len(self.demo_ini_state)))
        self.ref_traj = list()

        # ------------------------------ other setup ------------------------------
        self.iteration = 1
        self.Loss_his = []  # Original loss without barriers
        self.theta_error = []
        self.data_time = []
        self.gradient_time = []
        self.ekf_time = []
        self.x_his = []
        self.u_his = []
        self.theta_his = []
        self.cost_his = []
        self.barrier_cost_his = []  # Barrier costs from getCost

    def initialize_theta(self, initial_theta):
        self.theta = initial_theta
        print('theta = ', self.theta)

    def set_iteration(self, iteration):
        self.iteration = iteration

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R

    def solve(self):
        updateTheta = EKF()
        for iter in range(self.iteration):
            for idx in range(self.demo_horizon):
                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if idx == 0 and iter == 0:
                    # traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                    traj = self.sysoc.solveBarrierOC(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                else:
                    traj = self.sysoc.solveBarrierOCRef(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref=self.ref_traj)
                    # Use previous theta as reference and P_prev as P_inv
                    # traj = self.sysoc.safeEKF_ocSolver(
                    #     ini_state=self.demo_ini_state, 
                    #     horizon=self.demo_horizon, 
                    #     auxvar_value=self.theta,
                    #     theta_ref=self.theta_his[-2],  # Previous theta value
                    #     P_inv=np.linalg.inv(self.P_prev),  # Inverse of current P matrix
                    #     ref=self.ref_traj
                    # )
                    # traj2 = self.sysoc.solveBarrierOCRef(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref=self.ref_traj)
                    # print(traj['state_traj_opt'])
                    # print(traj2['state_traj_opt'])

                    # self.plotTraj(traj['state_traj_opt'], traj['control_traj_opt'])
                
                    
                # getCost(traj['state_traj_opt'], traj['control_traj_opt'], self.theta)

                self.ref_traj = traj
                
                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                aux_sol = self.sysoc.auxSysBarrierOC(opt_sol=traj)
                self.gradient_time += [time.time()-gradient_start_time]

                ekf_start_time = time.time()
                
                if aux_sol is not None:
                    # take solution of the auxiliary control system
                    dxdtheta_traj = aux_sol['state_traj_opt']
                    dudtheta_traj = aux_sol['control_traj_opt']

                    dxdtheta_t = dxdtheta_traj[idx]
                    dudtheta_t = dudtheta_traj[idx]

                    # --------------------------- Loss function, dLdXi ---------------------------------------- 
                    state_traj = traj['state_traj_opt']
                    control_traj = traj['control_traj_opt']

                    xi = SX.sym("xi", self.dynsys.X.shape[0] + self.dynsys.U.shape[0])

                    # Create demo trajectory data
                    demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
                    current_traj = np.hstack((state_traj[idx], control_traj[idx]))
                    current_state = (state_traj[idx])
                    current_control = (control_traj[idx])
                    
                    # Build gradient matrix
                    dxidtheta_t = np.vstack((dxdtheta_traj[idx], dudtheta_traj[idx]))

                    # loss = norm_2(demo_traj - xi)**2
                    loss = demo_traj - xi

                    barrier = 0

                    # if self.task_const_fn is not None:
                    #     task_const_sym = self.task_const_fn(xi[:self.dynsys.X.shape[0]], xi[self.dynsys.X.shape[0]:])
                    #     # Add barrier terms for each task constraint
                    #     for i in range(self.task_const_num):
                    #         # loss = loss + 1/self.beta /self.alpha* log(1 + exp(self.beta * self.task_const_fn[i]))
                    #         loss = loss + 1/self.beta /self.alpha* log(1 + exp(self.beta * task_const_sym[i]))

                    dLdXi = jacobian(loss, xi)
                    # dBarrierdXi = jacobian(barrier, xi)

                    lossFun = Function("lossFun", [xi], [loss])
                    # barrierFun = Function("barrierFun", [xi], [barrier])
                    dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])
                    # dBarrierdXiFun = Function("dBarrierdXiFun", [xi], [dBarrierdXi])

                    lossNow = lossFun(current_traj).full()
                    # barrierNow = barrierFun(current_traj).full()
                    # print(lossNow)
                    # print(barrierNow)
                    dLdXiNow = dLdXiFun(current_traj).full()
                    # dBarrierdXiNow = dBarrierdXiFun(current_traj).full()

                    # --------------------------- Chain rule ----------------------------------------
                    dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)
                    dp = dLdtheta

                else:
                    self.dp = np.zeros(self.theta.shape)
                    lossNow = np.zeros((self.dynsys.X.shape[0] + self.dynsys.U.shape[0], 1))


                self.evaluateLoss(state_traj, control_traj)

                if self.plotTrajFlag:
                    self.plotTraj(state_traj, control_traj)
                
               

                if self.iteration < 100:
                    print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                    print('theta = ', self.theta)
                else:
                    if(iter*self.demo_horizon+idx) % 100 == 0:
                        print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                
                # --------------------------- EKF ----------------------------------------
                # updateTheta = EKF()
                updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
                updateTheta.update(dp, self.R, lossNow)
                self.ekf_time += [time.time()-ekf_start_time]
                self.P_prev = updateTheta.P
                # print('P = ', np.linalg.norm(self.P_prev))
                # print('R = ', np.linalg.norm(self.Q_prev))
                self.theta = updateTheta.theta
                self.theta = np.where(self.theta < 1e-4, 1e-4, self.theta)
                self.data_time += [time.time()-data_start_time]
                self.x_his += [state_traj]
                self.u_his += [control_traj]
                self.theta_his += [self.theta]
                print('Time = ' + str(time.time()-data_start_time))

                # self.plotTraj(state_traj, control_traj)


        # # --------------------------- learned full iter ---------------------------
        # traj = self.sysoc.solveBarrierOCRef(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref=self.ref_traj)
        # state_traj = traj['state_traj_opt']
        # control_traj = traj['control_traj_opt']
        # self.evaluateLoss(state_traj, control_traj)
        # self.x_his += [state_traj]
        # self.u_his += [control_traj]

        # --------------------------- save all Loss ---------------------------
        self.plotTraj(state_traj, control_traj)
        if self.saveFlag:
            self.saveAll()
        
        self.plotLoss()

    def evaluateLoss(self, state_traj, control_traj):
        Loss = 0
        loss_his = []
        
        for jdx in range(self.demo_horizon):
            lossNorm = norm_2(state_traj[jdx]-self.demo_state_traj_original[jdx])**2 + norm_2(control_traj[jdx]-self.demo_control_traj[jdx])**2
            loss_his += [lossNorm]
            Loss += lossNorm
        
        self.Loss_his += [np.asarray(Loss)[0,0]]

    def getCost(self, state_traj, control_traj, theta):
        cost = 0
        barrier_sum = 0
        
        for idx in range(self.demo_horizon):
            # Original cost
            cost += self.sysoc.path_cost_fn(state_traj[idx], control_traj[idx], theta)
            
            # Path inequality barriers
            if hasattr(self.sysoc, 'path_inequ_cstr') and self.sysoc.path_inequ_cstr is not None:
                path_inequ_values = self.sysoc.path_inequ_cstr_fn(state_traj[idx], control_traj[idx], theta).full().flatten()
                for k in range(len(path_inequ_values)):
                    barrier_sum += 1/self.beta /self.alpha* log(1 + exp(self.beta * path_inequ_values[k]))
                    # print(path_inequ_values[k])
            
            # Path equality barriers
            if hasattr(self.sysoc, 'path_equ_cstr') and self.sysoc.path_equ_cstr is not None:
                path_equ_values = self.sysoc.path_equ_cstr_fn(state_traj[idx], control_traj[idx], theta).full().flatten()
                for k in range(len(path_equ_values)):
                    barrier_sum += 0.5 / self.gamma * (path_equ_values[k])**2
        
        # Final cost
        cost += self.sysoc.final_cost_fn(state_traj[-1], theta)
        
        # Final inequality barriers
        if hasattr(self.sysoc, 'final_inequ_cstr') and self.sysoc.final_inequ_cstr is not None:
            final_inequ_values = self.sysoc.final_inequ_cstr_fn(state_traj[-1], theta).full().flatten()
            for k in range(len(final_inequ_values)):
                barrier_sum += 1/self.beta /self.alpha* log(1 + exp(self.beta * final_inequ_values[k]))
        
        # Final equality barriers
        if hasattr(self.sysoc, 'final_equ_cstr') and self.sysoc.final_equ_cstr is not None:
            final_equ_values = self.sysoc.final_equ_cstr_fn(state_traj[-1], theta).full().flatten()
            for k in range(len(final_equ_values)):
                barrier_sum += 0.5 / self.gamma * (final_equ_values[k])**2
        
        return cost, barrier_sum
        

    def saveEach(self, idx, traj, loss_his):
        sio.savemat(self.dir+"results/iter_"+str(idx)+".mat", {'trajectories': traj,
                                                                'losses': loss_his,
                                                                'dt': self.dt,
                                                                'theta': self.theta})

    def saveAll(self):
        
        sio.savemat(self.dir+"results/results_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'Loss': self.Loss_his,
                                                  'SOCIL_time': self.data_time, 'Gradient_time': self.gradient_time,
                                                    'Estimator_time': self.ekf_time,
                                                    'true_theta': self.true_theta, 'theta': self.theta_his,
                                                    'demo_state': self.demo_state_traj, 'demo_control': self.demo_control_traj,
                                                    'demo_state_original': self.demo_state_traj_original,
                                                    'state': self.x_his, 'control': self.u_his})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        # Plot original loss (demo trajectory tracking)
        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.project)

        
        # Add dashed line for theoretical maximum barrier value
        # if self.task_const_fn is not None:
        #     num_task_constraints = self.task_const_fn.numel_out()
        #     max_barrier_value = num_task_constraints * self.gamma/self.beta * log(1 + exp(self.beta * 0))
        #     axs.axhline(y=max_barrier_value, color='r', linestyle='--', label=f'Max barrier: {max_barrier_value:.4f}')
        #     axs.legend()

        # Plot cost
        # fig, axs = plt.subplots()
        # axs.plot(self.cost_his)
        # # axs.axhline(self.demo_cost, color='r', linestyle='--')
        # axs.set_xlabel("Data")
        # axs.set_ylabel("Cost")
        # axs.set_title(self.project + " - Cost")
        
        # # Plot barrier costs from getCost
        # fig, axs = plt.subplots()
        # axs.plot(self.barrier_cost_his)
        # plt.yscale("log")
        # axs.set_xlabel("Data")
        # axs.set_ylabel("Cost")
        # axs.set_title(self.project + " - Barrier Cost J")
        plt.show()

    def plotTraj(self, state_traj, control_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx], 'b')
            axs[idx].plot(iter, self.demo_state_traj[:,idx], 'r')
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj, 'b')
            axs.plot(iter, self.demo_control_traj, 'r')
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx], 'b')
                axs[idx].plot(iter, self.demo_control_traj[:,idx], 'r')
                axs[idx].set_ylabel("u"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")

        if self.project == "Quadrotor":
            # make 2 2D plots
            fig, axs = plt.subplots()
            axs.plot(self.demo_state_traj[:,0], self.demo_state_traj[:,1], 'r')
            axs.plot(self.demo_state_traj[-1,0], self.demo_state_traj[-1,1], 'r*')
            axs.plot(state_traj[:,0], state_traj[:,1], 'b')
            if self.task_const_fn is not None:
                # plot the circle
                circle = plt.Circle(self.dynsys.obstacle_pos, self.dynsys.obstacle_radius, color='gray', fill=True)
                axs.add_artist(circle)
            axs.set_xlabel("x")
            axs.set_ylabel("y")
            axs.set_title("2 D Trajectory")
            axs.legend(["Demo Trajectory", "Target", "Trajectory"])


            # Create 3D plot
            fig = plt.figure()
            axs = fig.add_subplot(111, projection='3d')

            # Plot demonstration trajectory in red
            axs.plot(self.demo_state_traj[:,0], self.demo_state_traj[:,1], self.demo_state_traj[:,2], 'r')

            # Plot final demo state as red star
            axs.plot([self.demo_state_traj[-1,0]], [self.demo_state_traj[-1,1]], [self.demo_state_traj[-1,2]], 'r*')

            # Plot optimized state trajectory in blue
            axs.plot(state_traj[:,0], state_traj[:,1], state_traj[:,2], 'b')

            # Plot obstacle as a 3D cylinder if obstacle position and radius are defined
            if self.task_const_fn is not None:
                # Example cylinder centered at (x, y) with radius r, extending in z direction
                center = self.dynsys.obstacle_pos
                radius = self.dynsys.obstacle_radius

                # Generate a 3D cylinder
                theta = np.linspace(0, 2 * np.pi, 30)
                z = np.linspace(0, 10, 2)  # Extend from z=0 to z=10
                theta_grid, z_grid = np.meshgrid(theta, z)
                x = center[0] + radius * np.cos(theta_grid)
                y = center[1] + radius * np.sin(theta_grid)
                axs.plot_surface(x, y, z_grid, color='gray', alpha=0.5)

            # Labels
            axs.set_xlabel("x")
            axs.set_ylabel("y")
            axs.set_zlabel("z")
            axs.set_zlim(-1,1)

            plt.show()


        plt.show()





