#!/usr/bin python3

import numpy as np
import array

# from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
import casadi as ca
import utils.use_env as use_env


class ARPAE_Planner():
    def __init__(self, sym_q, sym_u, sym_e, model):
        self.dt = 0.05
        self.nx = sym_q.shape[0]
        self.nu = sym_u.shape[0]
        self.OCP_N = sym_u.shape[1]
        self.stop = []

        self.model = model
        self.q = sym_q # state
        self.u = sym_u # control
        self.e = sym_e # energy
        self.road_curvature = np.zeros((1,self.OCP_N))

    def model_equality_constraint_k(self, qkp1, qk, uk, road_curvature):
        equality_constraint = qkp1 - self.model.modelconstraint(road_curvature,qk,uk)
        return equality_constraint

    def model_equality_constraints(self):
        con = self.model_equality_constraint_k(self.q[:,1], self.q[:,0], self.u[:,0], self.road_curvature[0,0])
        for k in range(1, self.OCP_N):
            k_con = self.model_equality_constraint_k(self.q[:,k+1], self.q[:,k], self.u[:,k], self.road_curvature[0,k])
            con = ca.vertcat(con, k_con)

        bg = np.zeros(con.shape)
        return con, bg

    def checker(self, x0, u0, K, ref):
        ey_flag = True
        ephi_flag = True
        vx_flag = True
        k_flag = True

        ey_flag = x0[1,0] <= ref[3]+1 and x0[1,0] >= ref[3]-1
        ephi_flag = x0[2,0] <= np.pi/15 and x0[2,0] >= -np.pi/15
        # vx_flag = u0[0,0] - self.ref[0].item(0) <= 1 and u0[0,0] - self.ref[0].item(0) >= -1
        # k_flag = u0[1,0] - K.item(0) <= 0.01 and u0[1,0] - K.item(0) >= -0.01
        flag = ey_flag and ephi_flag and vx_flag and k_flag
        return flag

    def initialize(self, x0, u0, K, strategy_cur, npc_states, tr_lights, vline_tar, vline_cur, mode, mat_bnd, ref):
        if mode == 'LC':
            self.x0 = x0
            self.x0[0,0] = x0[0,0] + u0[0,0]*0.5
        else:
            self.x0 = x0
        # self.x0 = x0 # num of state : [station(s), lateral error(ey), heading error(ephi)]
        self.u0 = u0 # num of control : [longitudinal speed[vx], curvature[k]]
        self.road_curvature = K # should be 1 X MPC_N array
        self.mat_bnd = mat_bnd
        self.ref = ref
        self.LC_LK_mode = mode
        #[0: v_ref, 1: s_start, 2: s_end, 3: ey_target_ref, 4: ey_current_ref]
        # s_start ~ s_end (freespace) @ terminal time
        # ey_target_ref : target lane @ terminal time
        # ey_current_ref : current lane

        '''
        terminal constraint for s
        '''
        s_min = self.ref[1]
        s_max = self.ref[2]

        '''
        ey_boundary calculation
        '''
        eylimit = 3
        ey_min = np.amin([self.ref[4]-eylimit, self.ref[3]-eylimit])
        ey_max = np.amax([self.ref[4]+eylimit, self.ref[3]+eylimit])

        '''
        Initial condition for motion planning
        '''
        initial_con = ca.vertcat(self.q[:,0] - self.x0, self.u[:,0] - self.u0)
        initial_bg = np.zeros(initial_con.shape)

        # Initialize the constraint. allcon
        all_con = initial_con
        self.all_ubg = initial_bg
        self.all_lbg = initial_bg

        '''
        Vehicle dynamics constraints
        '''
        try:
            model_constraints, model_bg = self.model_equality_constraints()
            all_con = ca.vertcat(all_con, model_constraints)
            self.all_ubg = ca.vertcat(self.all_ubg, model_bg)
            self.all_lbg = ca.vertcat(self.all_lbg, model_bg)
        except:
            print('No model constraint')

        '''
        State box constraint: |ephi|<=pi/2, 0<=vx<=15, ey_min<=ey<=ey_max
        '''
        state_box_con = ca.vertcat(self.q[1,0] - ey_min, ey_max - self.q[1,0], self.q[2,0] + np.pi/2, np.pi/2 - self.q[2,0], 15 - self.u[0,0], self.u[0,0])
        for i in range(1,self.OCP_N):
            state_box_con = ca.vertcat(state_box_con, self.q[1,i] - ey_min, ey_max - self.q[1,i], self.q[2,i] + np.pi/2, np.pi/2 - self.q[2,i], 15 - self.u[0,i], self.u[0,i])

        state_box_ubg = np.inf * np.ones(state_box_con.shape)
        state_box_lbg = np.zeros(state_box_con.shape)

        # state box constraint update
        all_con = ca.vertcat(all_con, state_box_con)
        self.all_ubg = ca.vertcat(self.all_ubg, state_box_ubg)
        self.all_lbg = ca.vertcat(self.all_lbg, state_box_lbg)

        '''
        ay limit constraint
        '''
        ay_con = ca.vertcat(self.u[0,0]**2*self.u[1,0]+1.5, -self.u[0,0]**2*self.u[1,0]+1.5)
        for i in range(1,self.OCP_N):
            ay_con = ca.vertcat(ay_con, self.u[0,i]**2*self.u[1,i]+1.5, -self.u[0,i]**2*self.u[1,i]+1.5)

        ay_ubg = np.inf * np.ones(ay_con.shape)
        ay_lbg = np.zeros(ay_con.shape)

        # state box constraint update
        all_con = ca.vertcat(all_con, ay_con)
        self.all_ubg = ca.vertcat(self.all_ubg, ay_ubg)
        self.all_lbg = ca.vertcat(self.all_lbg, ay_lbg)


        '''
        Terminal constraint
        '''
        terminal_con = ca.vertcat(self.q[0,-1] - s_min, s_max - self.q[0,-1], self.ref[3] + 0.5 - self.q[1,-1], -self.ref[3] + 0.5 + self.q[1,-1], self.u[0,-1] + 1 - self.ref[0][0,-1], -self.u[0,-1] + 1 + self.ref[0][0,-1])
        terminal_ubg = np.inf * np.ones(terminal_con.shape)
        terminal_lbg = np.zeros(terminal_con.shape)

        all_con = ca.vertcat(all_con, terminal_con)
        self.all_ubg = ca.vertcat(self.all_ubg, terminal_ubg)
        self.all_lbg = ca.vertcat(self.all_lbg, terminal_lbg)

        '''
        Objective functions
        '''
        obj = 0
        v_maintain = 5
        gamma = 1 # coefficient corresponds to the terminal constraints

        if self.LC_LK_mode == 'LC':
            print("mode : LC")
            '''
            OBCA (will be updated)
            '''
            # self.use_env = use_env.USE_ENV(self.x0, self.OCP_N)
            # self.use_env.update(strategy_cur)
            # self.use_env.observe(npc_states, tr_lights, vline_tar, vline_cur)
            # front_npc, next_npcs = self.use_env.grouping_npc()
            #
            # lf = 1.5 + 5 # saftey distance 5m
            # tw = 1.5
            # lr = 1.5 + 5
            # if len(front_npc):
            #     s_pred_1 = front_npc[1] * np.arange(self.OCP_N) * Ts + front_npc[0]
            #     b_t_1 = np.vstack((s_pred_1, s_pred_1 * (-1), s_pred_1 * 0 + front_npc[2], s_pred_1 * 0 - front_npc[2]))
            #     A_m_1 = np.array([[1,0],[-1,0],[0,1],[0,-1]])
            #     b_m_1 = np.array([[lf],[lr],[tw/2],[tw/2]])

            # self.obca_lambda = ca.MX.sym('obca_lambda', 4, self.OCP_N)

            # if len(next_npcs):
            #     # test take only first two of next_npcs
            #     cnt =
            #     for item in next_npcs:

            #obstacle 1
            # obca_con = ca.vertcat((A_m_1[:,0]*q[0,0] + A_m_1[:,1]*q[1,0] - b_m_1[:,0] - b_t_1[:,0]).T @ self.obca_lambda[:,0] - 1, -(A_m_1.T @ self.obca_lambda[:,0]).T @ (A_m_1.T @ self.obca_lambda[:,0]) + 1)
            # for i in range(1,self.OCP_N):
            #     obca_con = ca.vertcat(obca_con, (A_m_1[:,0]*q[0,i] + A_m_1[:,1]*q[1,i] - b_m_1[:,0] - b_t_1[:,i]).T @ self.obca_lambda[:,i] - 1, -(A_m_1.T @ self.obca_lambda[:,i]).T @ (A_m_1.T @ self.obca_lambda[:,i]) + 1)
            #
            # for i in range(self.OCP_N):
            #     for j in range(4):
            #         obca_con = ca.vertcat(obca_con, self.obca_lambda[j,i])
            #
            # obca_ubg = np.inf * np.ones(obca_con.shape)
            # obca_lbg = np.zeros(obca_con.shape)
            #
            # all_con = ca.vertcat(all_con, obca_con)
            # self.all_ubg = ca.vertcat(self.all_ubg, obca_ubg)
            # self.all_lbg = ca.vertcat(self.all_lbg, obca_lbg)

            # opti.subject_to((A_m_1[:,0]*s[k] + A_m_1[:,1]*ey[k] - b_m_1[:,0] - b_t_1[:,k]).T @ lamb_1[:,k] >= 1)
            # opti.subject_to((A_m_1.T @ lamb_1[:,k]).T @ (A_m_1.T @ lamb_1[:,k]) <= 1)
            # for j in range(4):
            #     opti.subject_to(lamb_1[j,k] >= 0)

            # terminal condition
            # opti.subject_to(s[-1] >= s_pred_2[-1])
            # opti.subject_to(ey[-1] >= 3)
            # opti.subject_to(ey[-1] <= 4)



            obj += gamma * ((self.q[1,-1] - self.ref[3])**2 + self.q[2,-1]**2 + self.u[1,-1]**2)
            for i in range(self.OCP_N-2):
                obj += 1*(self.u[1,i] - 2*self.u[1,i+1] + self.u[1,i+2])**2
                obj += 1*(self.u[1,i] - self.u[1,i+1])**2
                obj += (self.u[0,i] - 2*self.u[0,i+1] + self.u[0,i+2])**2
                obj += (self.u[0,i] - self.u[0,i+1])**2

            nlp = {'x':ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e)), 'f': obj, 'g': all_con}
            plugin_opts = {'verbose': False, 'verbose_init': False, 'print_time': False, 'ipopt': {"print_level": 0}}
            self.S = ca.nlpsol('S', 'ipopt', nlp, plugin_opts)

        elif self.LC_LK_mode == 'STOP':
            print("mode : STOP")
            '''
            Stop constraint
            '''
            self.stop_slack = ca.MX.sym('stop_slack', 2, self.OCP_N)
            self.terminal_stop_slack = ca.MX.sym('terminal_stop_slack', 1, 1)
            max_ax = -3
            stop_vx_upper_bound = 2*max_ax*(self.q[0,0] - self.stop)
            stop_con = ca.vertcat(self.u[0,0] + self.stop_slack[0,0], -self.u[0,0]**2 + stop_vx_upper_bound + self.stop_slack[1,0], self.stop_slack[0,0], self.stop_slack[1,0])
            for i in range(1,self.OCP_N):
                stop_vx_upper_bound = 2*max_ax*(self.q[0,i] - self.stop)
                stop_con = ca.vertcat(stop_con,self.u[0,i] + self.stop_slack[0,i], -self.u[0,i]**2 + stop_vx_upper_bound + self.stop_slack[1,i], self.stop_slack[0,i], self.stop_slack[1,i])

            stop_con = ca.vertcat(stop_con,-self.q[0,-1] + self.terminal_stop_slack[0,0] + self.stop, self.terminal_stop_slack[0,0])

            stop_ubg = np.inf * np.ones(stop_con.shape)
            stop_lbg = np.zeros(stop_con.shape)

            # state box constraint update
            all_con = ca.vertcat(all_con, stop_con)
            self.all_ubg = ca.vertcat(self.all_ubg, stop_ubg)
            self.all_lbg = ca.vertcat(self.all_lbg, stop_lbg)

            slack_weighting = 100
            obj += gamma * ((self.q[1,-1] - self.ref[3])**2 + self.q[2,-1]**2 + self.u[1,-1]**2) + slack_weighting * self.terminal_stop_slack[0,0]
            for i in range(self.OCP_N-2):
                obj += 1*(self.u[1,i] - 2*self.u[1,i+1] + self.u[1,i+2])**2
                obj += 1*(self.u[1,i] - self.u[1,i+1])**2
                obj += (self.u[0,i] - 2*self.u[0,i+1] + self.u[0,i+2])**2
                # obj += (self.u[0,i] - self.u[0,i+1])**2

            # slacks
            for i in range(self.OCP_N):
                obj += slack_weighting * self.stop_slack[0,i] + slack_weighting * self.stop_slack[1,i]

            nlp = {'x':ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e), ca.vec(self.stop_slack), ca.vec(self.terminal_stop_slack)), 'f': obj, 'g': all_con}
            plugin_opts = {'verbose': False, 'verbose_init': False, 'print_time': False, 'ipopt': {"print_level": 0}}
            self.S = ca.nlpsol('S', 'ipopt', nlp, plugin_opts)

        else:
            print("mode : LK")
            '''
            Energy constraint
            '''
            # energy_con = ca.vertcat(self.e[0,0] - (self.u[0,1]**2 - self.u[0,0]**2), self.e[0,0])
            # for i in range(1,self.OCP_N-1):
            #     energy_con = ca.vertcat(energy_con, self.e[0,i] - (self.u[0,i+1]**2 - self.u[0,i]**2), self.e[0,i])
            #
            # energy_ubg = np.inf * np.ones(energy_con.shape)
            # energy_lbg = np.zeros(energy_con.shape)
            #
            # # energy constraint update
            # all_con = ca.vertcat(all_con, energy_con)
            # self.all_ubg = ca.vertcat(self.all_ubg, energy_ubg)
            # self.all_lbg = ca.vertcat(self.all_lbg, energy_lbg)

            # obj += v_maintain * (self.u[0,-1] - self.ref[0][0,-1])**2
            obj += gamma * ((self.q[1,-1] - self.ref[3])**2 + self.q[2,-1]**2 + self.u[1,-1]**2)
            for i in range(self.OCP_N-2):
                obj += 1*(self.u[1,i] - 2*self.u[1,i+1] + self.u[1,i+2])**2
                obj += 1*(self.u[1,i] - self.u[1,i+1])**2
                obj += (self.u[0,i] - 2*self.u[0,i+1] + self.u[0,i+2])**2
                obj += (self.u[0,i] - self.u[0,i+1])**2

            # tracking cost in planner
            for i in range(self.OCP_N):
                obj += v_maintain * (self.u[0,i] - self.ref[0][0,i])**2

            nlp = {'x':ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e)), 'f': obj, 'g': all_con}
            plugin_opts = {'verbose': False, 'verbose_init': False, 'print_time': False, 'ipopt': {"print_level": 0}}
            self.S = ca.nlpsol('S', 'ipopt', nlp, plugin_opts)


    def lclkmode_feedback(self, x0, u0, K, ref):
        flag = self.checker(x0, u0, K, ref) # if LC is finished, flag is True
        cur_lane = self.ref[4]
        target_lane = self.ref[3]
        if self.LC_LK_mode == 'LC':
            if flag:
                self.LC_LK_mode = 'LK'
                cur_lane = target_lane
        return self.LC_LK_mode, cur_lane, target_lane
    def stopmode_check(self, x0, u0, stop_list, t_safety):
        gapmin = 1000
        closest_stop = []
        # find the closest stop from the ego
        for stop in stop_list:
            if stop - x0[0,0] <= gapmin and stop - x0[0,0] >=0:
                closest_stop = stop
                gapmin = stop - x0[0,0]
        # decide mode
        if closest_stop - x0[0,0] <= t_safety*u0[0,0]: # remaing_dist <= safe distance
            mode = 'STOP'
        else:
            mode = 'LK'

        self.stop = closest_stop
        return closest_stop, mode

    def solve(self):

        if self.LC_LK_mode == 'LC':
            r = self.S(x0=np.zeros(ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e)).shape), lbg= self.all_lbg, ubg= self.all_ubg)
            x_opt = r['x']
        elif self.LC_LK_mode == 'STOP':
            r = self.S(x0=np.zeros(ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e), ca.vec(self.stop_slack), ca.vec(self.terminal_stop_slack)).shape), lbg= self.all_lbg, ubg= self.all_ubg)
            x_opt = r['x']
        else:
            r = self.S(x0=np.zeros(ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e)).shape), lbg= self.all_lbg, ubg= self.all_ubg)
            x_opt = r['x']

        # state reconstruction
        for i in range(self.OCP_N + 1):
            if i < 1:
                state = np.array(x_opt[0:self.nx])
            else:
                state = np.hstack((state, np.array(x_opt[self.nx * i:self.nx * (i+1)])))


        control_start_ind = self.nx * (self.OCP_N + 1)
        # control input reconstruction
        for i in range(self.OCP_N):
            if i < 1:
                control = np.array(x_opt[control_start_ind:control_start_ind+self.nu])
            else:
                control = np.hstack((control, np.array(x_opt[control_start_ind + self.nu * i:control_start_ind + self.nu * (i+1)])))


        energy_start_ind = self.nx * (self.OCP_N + 1) + self.nu * (self.OCP_N)
        # energy reconstruction
        for i in range(self.OCP_N):
            if i < 1:
                energy = np.array(x_opt[energy_start_ind:energy_start_ind+1])
            else:
                energy = np.hstack((energy, np.array(x_opt[energy_start_ind + i:energy_start_ind + (i+1)])))

        return state, control, energy # np.array type. nx x mpc_N + 1, nu x mpc_N, 1 x mpc_N
