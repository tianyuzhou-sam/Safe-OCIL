import numpy as np
from casadi import *
import scipy.io as sio
from scipy.special import softmax
import matplotlib.pyplot as plt 
import os, shutil
import sys
import time
import copy
sys.path.append(os.getcwd() + '/src/')
import JinEnv
from EKF import EKF
from Loss_function import Loss
import ocSolver


class ImitationLearning:
    def __init__(self, project="", mode="", dynsys=None, noise=None, gamma=1e-2, dir="", demoFile="", saveFlag=False):

        if not (mode == "Objective" or mode == "Dynamic" or mode == "All"):
            print("Mode not defined!")
            sys.exit()

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

        print(data['true_parameter'].flatten())
        print(self.true_theta)
        self.true_theta = np.hstack((self.true_theta, [10]))

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
        self.sysoc.convert2BarrierOC(gamma=gamma)

        # ------------------------------ initilize tunable parameter ------------------------------
        self.sigma = 0.9
        self.initial_theta = self.true_theta + self.true_theta * (self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2)
        self.theta = self.initial_theta
        print('theta = ', self.theta)

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj_original = self.trajectories[0, 0]['state_traj_opt']
        self.demo_control_traj = self.trajectories[0, 0]['control_traj_opt']
        self.demo_ini_state = self.demo_state_traj_original[0, :]
        self.demo_horizon = self.demo_control_traj.shape[0]
        self.demo_cost = self.getCost(self.demo_state_traj_original, self.demo_control_traj, self.true_theta)

        ocTraj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.true_theta)        
        self.demo_cost = ocTraj['cost'] 
        # if project == 'uniform':
        # self.demo_state_traj = self.demo_state_traj_original + (np.random.random((self.demo_horizon+1,len(init_state)))-0.5)*noise
        # if project == 'normal':
        self.demo_state_traj = self.demo_state_traj_original + np.random.normal(0, noise, (self.demo_horizon+1,len(self.demo_ini_state)))
        self.ref_traj = list()

        # ------------------------------ other setup ------------------------------
        self.iteration = 1
        self.Loss_his = []
        self.theta_error = []
        self.data_time = []
        self.gradient_time = []
        self.ekf_time = []
        self.x_his = []
        self.u_his = []
        self.theta_his = [self.theta]
        self.cost_his = []

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.initial_theta = self.true_theta + self.true_theta * (self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2)
        self.theta = self.initial_theta
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
            for idx in range(1, self.demo_horizon):
                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if idx == 1:
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

                    # fig, axs = plt.subplots(4,1)
                    # for i in range(4):
                    #     axs[i].plot(traj['state_traj_opt'][:,i])
                    #     # axs[i].plot(traj2['state_traj_opt'][:,i])
                    #     axs[i].plot(self.demo_state_traj[:,i])
                    # plt.show()
                    


                self.ref_traj = traj
                
                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                aux_sol = self.sysoc.auxSysBarrierOC(opt_sol=traj)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']

                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]

                # --------------------------- Loss function, dLdXi ---------------------------------------- 
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']

                # xi = SX.sym("xi", self.dynsys.X.shape[0]+self.dynsys.U.shape[0])
                # demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
                # current_traj = np.hstack((state_traj[idx], control_traj[idx]))
                # dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

                xi = SX.sym("xi", self.dynsys.X.shape[0])
                demo_traj = (self.demo_state_traj[idx])
                current_traj = (state_traj[idx])
                dxidtheta_t = dxdtheta_t

                epsilon = 1e0
                factor = 1e0

                C = -xi[2]-4
                C_value = -current_traj[2]-4

                K = 1
                
                loss = demo_traj - xi
                # print('------------', np.log(1+np.exp(C_value)))
                loss = demo_traj - xi + epsilon * np.log(1+np.exp(C))
                # barrier = 0
                # if C_value <= 0:
                #     barrier = 0
                # else:
                #     barrier = C**2 * (C/(np.sqrt(C**2+epsilon)))**K
                # loss = demo_traj - xi + factor * barrier
                dLdXi = jacobian(loss, xi)

                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()

                self.evaluateLoss(state_traj, control_traj)

                
                if self.plotTrajFlag:
                    self.plotTraj(state_traj, control_traj)

                # evaluate the loss
                dldx_traj = state_traj - self.demo_state_traj
                dldu_traj = control_traj - self.demo_control_traj
                
                # --------------------------- Chain rule ----------------------------------------
                dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)
                dp = dLdtheta

                if self.iteration < 10:
                    print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                    print('theta = ', self.theta)
                else:
                    if(iter*self.demo_horizon+idx) % 100 == 0:
                        print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                
                # --------------------------- EKF ----------------------------------------
                ekf_start_time = time.time()
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
                self.cost_his += [self.getCost(state_traj, control_traj, self.theta)]
                print('Time = ' + str(time.time()-data_start_time))

                # self.plotTraj(state_traj, control_traj)


        # --------------------------- learned full iter ---------------------------
        # traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
        # traj = self.sysoc.solveBarrierOC(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
        traj = self.sysoc.solveBarrierOCRef(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref=self.ref_traj)
        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']
        self.evaluateLoss(state_traj, control_traj)
        self.x_his += [state_traj]
        self.u_his += [control_traj]
        self.cost_his += [self.getCost(state_traj, control_traj, self.theta)]
        self.cost_his = np.array(self.cost_his).squeeze()

        # --------------------------- save all Loss ---------------------------
        self.plotTraj(state_traj, control_traj)
        if self.saveFlag:
            self.saveAll()
        
        self.plotLoss()

    def evaluateLoss(self, state_traj, control_traj):
        Loss = 0
        loss_his = []
        for jdx in range(self.demo_horizon):
            each_traj_t = np.hstack((state_traj[jdx], control_traj[jdx]))
            demo_traj_t = np.hstack((self.demo_state_traj_original[jdx], self.demo_control_traj[jdx]))
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        self.Loss_his += [np.asarray(Loss)[0,0]]
        self.theta_error += [np.asarray(norm_2(self.theta-self.true_theta)**2)[0,0]]

    def getCost(self, state_traj, control_traj, theta):
        cost = 0
        for idx in range(self.demo_horizon):
            cost += self.sysoc.path_cost_fn(state_traj[idx], control_traj[idx], theta)
        cost += self.sysoc.final_cost_fn(state_traj[-1], theta)
        return cost
        

    def saveEach(self, idx, traj, loss_his):
        sio.savemat(self.dir+"results/iter_"+str(idx)+".mat", {'trajectories': traj,
                                                                'losses': loss_his,
                                                                'dt': self.dt,
                                                                'theta': self.theta})

    def saveAll(self):
        
        sio.savemat(self.dir+"results/Loss_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'Loss': self.Loss_his,
                                                  'theta': self.theta_error})
        sio.savemat(self.dir+"results/time.mat", {'Data': self.data_time, 'Gradient': self.gradient_time,
                                                    'EKF': self.ekf_time})
        sio.savemat(self.dir+"results/theta_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'true_theta': self.true_theta, 'theta': self.theta_his,
                                                    'demo_state': self.demo_state_traj, 'demo_control': self.demo_control_traj,
                                                    'state': self.x_his, 'control': self.u_his})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.mode + ": " + self.project)

        fig, axs = plt.subplots()
        axs.plot(self.theta_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Theta Error")
        axs.set_title(self.mode + ": " + self.project)

        fig, axs = plt.subplots()
        axs.plot(self.cost_his)
        axs.axhline(self.demo_cost, color='r', linestyle='--')
        axs.set_xlabel("Data")
        axs.set_ylabel("Cost")
        axs.set_title(self.mode + ": " + self.project)
        plt.show()

    def plotTraj(self, state_traj, control_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].plot(iter, self.demo_state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj)
            axs.plot(iter, self.demo_control_traj)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx])
                axs[idx].plot(iter, self.demo_control_traj[:,idx])
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()




class ConstrainedEKF:
    def __init__(self, project="", mode="", dynsys=None, noise=None, gamma=1e-2, dir="", demoFile="", saveFlag=False):

        if not (mode == "Objective" or mode == "Dynamic" or mode == "All"):
            print("Mode not defined!")
            sys.exit()

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

        print(data['true_parameter'].flatten())
        print(self.true_theta)
        self.true_theta = np.hstack((self.true_theta, [10]))

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
        # self.sysoc.convert2BarrierOC(gamma=gamma)

        # ------------------------------ initilize tunable parameter ------------------------------
        self.sigma = 0.9
        self.initial_theta = self.true_theta + self.true_theta * (self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2)
        self.theta = self.initial_theta
        print('theta = ', self.theta)

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj_original = self.trajectories[0, 0]['state_traj_opt']
        self.demo_control_traj = self.trajectories[0, 0]['control_traj_opt']
        self.demo_ini_state = self.demo_state_traj_original[0, :]
        self.demo_horizon = self.demo_control_traj.shape[0]
        self.demo_cost = self.trajectories[0, 0]['cost']
        # if project == 'uniform':
        # self.demo_state_traj = self.demo_state_traj_original + (np.random.random((self.demo_horizon+1,len(init_state)))-0.5)*noise
        # if project == 'normal':
        self.demo_state_traj = self.demo_state_traj_original + np.random.normal(0, noise, (self.demo_horizon+1,len(self.demo_ini_state)))
        self.ref_traj = list()

        # ------------------------------ other setup ------------------------------
        self.iteration = 1
        self.Loss_his = []
        self.theta_error = []
        self.data_time = []
        self.gradient_time = []
        self.ekf_time = []
        self.x_his = []
        self.u_his = []
        self.theta_his = [self.theta]
        self.cost_his = []

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.initial_theta = self.true_theta + self.true_theta * (self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2)
        self.theta = self.initial_theta
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
            for idx in range(1, self.demo_horizon):
                data_start_time = time.time()
                # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
                if True:
                    # traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                    traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
                else:
                    traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
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

                    # fig, axs = plt.subplots(4,1)
                    # for i in range(4):
                    #     axs[i].plot(traj['state_traj_opt'][:,i])
                    #     # axs[i].plot(traj2['state_traj_opt'][:,i])
                    #     axs[i].plot(self.demo_state_traj[:,i])
                    # plt.show()
                    


                self.ref_traj = traj
                
                # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
                gradient_start_time = time.time()
                auxsys = self.sysoc.getAuxSys(opt_sol=traj, threshold=1e-5)
                self.clqr.auxsys2Eqctlqr(auxsys=auxsys)
                aux_sol = self.clqr.eqctlqrSolver(threshold=1e-5)
                self.gradient_time += [time.time()-gradient_start_time]
                # take solution of the auxiliary control system
                dxdtheta_traj = aux_sol['state_traj_opt']
                dudtheta_traj = aux_sol['control_traj_opt']

                dxdtheta_t = dxdtheta_traj[idx]
                dudtheta_t = dudtheta_traj[idx]

                # --------------------------- Loss function, dLdXi ---------------------------------------- 
                state_traj = traj['state_traj_opt']
                control_traj = traj['control_traj_opt']

                # xi = SX.sym("xi", self.dynsys.X.shape[0]+self.dynsys.U.shape[0])
                # demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
                # current_traj = np.hstack((state_traj[idx], control_traj[idx]))
                # dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

                xi = SX.sym("xi", self.dynsys.X.shape[0])
                demo_traj = (self.demo_state_traj[idx])
                current_traj = (state_traj[idx])
                dxidtheta_t = dxdtheta_t


                loss = demo_traj - xi
                # print('------------', np.log(1+np.exp(C_value)))
                # loss = demo_traj - xi + epsilon * np.log(1+np.exp(C))
                dLdXi = jacobian(loss, xi)

                lossFun = Function("lossFun", [xi], [loss])
                dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

                lossNow = lossFun(current_traj).full()
                dLdXiNow = dLdXiFun(current_traj).full()

                self.evaluateLoss(state_traj, control_traj)

                
                if self.plotTrajFlag:
                    self.plotTraj(state_traj, control_traj)

                # evaluate the loss
                dldx_traj = state_traj - self.demo_state_traj
                dldu_traj = control_traj - self.demo_control_traj
                
                # --------------------------- Chain rule ----------------------------------------
                dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)
                dp = dLdtheta

                if self.iteration < 10:
                    print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                    print('theta = ', self.theta)
                else:
                    if(iter*self.demo_horizon+idx) % 100 == 0:
                        print('Data = ', iter*self.demo_horizon+idx, 'Loss = ', self.Loss_his[-1])
                
                # --------------------------- EKF ----------------------------------------
                ekf_start_time = time.time()
                # updateTheta = EKF()
                updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
                updateTheta.update(dp, self.R, lossNow)
                self.theta = updateTheta.theta


                epsilon = 1e0*2

                C = -xi[2]-4
                C_value = -current_traj[2]-4

                dCdXi = jacobian(C, xi)
                dCdXiFun = Function("dCdXiFun", [xi], [dCdXi])
                dCdXiNow = dCdXiFun(current_traj).full()

                num_task_c = 1
                

                theta = self.theta
                print(C_value)
                print(current_traj[2])

                for nC in range(num_task_c):
                    if C_value > 0:
                        dCdtheta = np.matmul(dCdXiNow, dxdtheta_t)
                        
                        # F = np.hstack((dCdtheta, np.eye(num_task_c)))
                        # G1 = np.hstack((-dLdtheta, np.zeros((len(self.demo_ini_state), num_task_c))))
                        # G2 = np.hstack((np.eye(len(theta)), np.zeros((len(theta), num_task_c))))
                        # G = np.vstack((G1, G2))
                    

                        b = -C_value + np.matmul(dCdtheta, theta)
                        # b = np.vstack((b, 0))


                        # theta = updateTheta.C_update(F, G, current_traj, b)
                        theta = updateTheta.C_update(dCdtheta, -dLdtheta, b)
                        exit()




                self.ekf_time += [time.time()-ekf_start_time]
                self.P_prev = updateTheta.P
                # print('P = ', np.linalg.norm(self.P_prev))
                # print('R = ', np.linalg.norm(self.Q_prev))
                
                self.theta = np.where(self.theta < 1e-4, 1e-4, self.theta)
                self.data_time += [time.time()-data_start_time]
                self.x_his += [state_traj]
                self.u_his += [control_traj]
                self.theta_his += [self.theta]
                self.cost_his += [traj['cost'][0]]
                print('Time = ' + str(time.time()-data_start_time))

                # self.plotTraj(state_traj, control_traj)


        # --------------------------- learned full iter ---------------------------
        traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
        # traj = self.sysoc.solveBarrierOC(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
        # traj = self.sysoc.solveBarrierOCRef(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta, ref=self.ref_traj)
        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']
        self.evaluateLoss(state_traj, control_traj)
        self.x_his += [state_traj]
        self.u_his += [control_traj]
        self.cost_his += [traj['cost'][0]]

        # --------------------------- save all Loss ---------------------------
        self.plotTraj(state_traj, control_traj)
        if self.saveFlag:
            self.saveAll()
        
        self.plotLoss()

    def evaluateLoss(self, state_traj, control_traj):
        Loss = 0
        loss_his = []
        for jdx in range(self.demo_horizon):
            each_traj_t = np.hstack((state_traj[jdx], control_traj[jdx]))
            demo_traj_t = np.hstack((self.demo_state_traj_original[jdx], self.demo_control_traj[jdx]))
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm
        self.Loss_his += [np.asarray(Loss)[0,0]]
        self.theta_error += [np.asarray(norm_2(self.theta-self.true_theta)**2)[0,0]]
        

    def saveEach(self, idx, traj, loss_his):
        sio.savemat(self.dir+"results/iter_"+str(idx)+".mat", {'trajectories': traj,
                                                                'losses': loss_his,
                                                                'dt': self.dt,
                                                                'theta': self.theta})

    def saveAll(self):
        
        sio.savemat(self.dir+"results/Loss_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'Loss': self.Loss_his,
                                                  'theta': self.theta_error})
        sio.savemat(self.dir+"results/time.mat", {'Data': self.data_time, 'Gradient': self.gradient_time,
                                                    'EKF': self.ekf_time})
        sio.savemat(self.dir+"results/theta_" + time.strftime("%Y%m%d%H%M%S") + ".mat", {'true_theta': self.true_theta, 'theta': self.theta_his,
                                                    'demo_state': self.demo_state_traj, 'demo_control': self.demo_control_traj,
                                                    'state': self.x_his, 'control': self.u_his})


    def load(self, dir):
        data = sio.loadmat(dir)

    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.Loss_his)
        plt.yscale("log")
        axs.set_xlabel("Data")
        axs.set_ylabel("Loss")
        axs.set_title(self.mode + ": " + self.project)

        fig, axs = plt.subplots()
        axs.plot(self.theta_error)
        axs.set_xlabel("Data")
        axs.set_ylabel("Theta Error")
        axs.set_title(self.mode + ": " + self.project)

        fig, axs = plt.subplots()
        axs.plot(self.cost_his)
        axs.axhline(self.demo_cost, color='r', linestyle='--')
        axs.set_xlabel("Data")
        axs.set_ylabel("Cost")
        axs.set_title(self.mode + ": " + self.project)
        plt.show()

    def plotTraj(self, state_traj, control_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].plot(iter, self.demo_state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj)
            axs.plot(iter, self.demo_control_traj)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx])
                axs[idx].plot(iter, self.demo_control_traj[:,idx])
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()




