#!/usr/bin python3

import numpy as np
import array

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

# should be updated
from mpclab_controllers_arpae.model import CasadiARPAE, EnergyConfig
import casadi as ca
import mpclab_controllers_arpae.use_env as use_env




class ARPAE_MPCParams():
    '''
    template that stores all parameters needed for the node as well as default values
    '''
    def __init__(self):
        self.dt = 0.1
        ## will add cost matrix

class ARPAE_MPC():
    def __init__(self, sym_q, sym_u, sym_e, model, control_params):
        self.dt = control_params.dt
        self.nx = sym_q.shape[0]
        self.nu = sym_u.shape[0]
        self.MPC_N = sym_u.shape[1]
        self.model = model
        self.q = sym_q # state
        self.u = sym_u # control
        self.e = sym_e # energy
        self.road_curvature = np.zeros((1,self.MPC_N))

    def model_equality_constraint_k(self, qkp1, qk, uk, road_curvature):
        equality_constraint = qkp1 - self.model.modelconstraint(road_curvature,qk,uk)
        return equality_constraint

    def model_equality_constraints(self):
        con = self.model_equality_constraint_k(self.q[:,1], self.q[:,0], self.u[:,0], self.road_curvature[0,0])
        for k in range(1, self.MPC_N):
            k_con = self.model_equality_constraint_k(self.q[:,k+1], self.q[:,k], self.u[:,k], self.road_curvature[0,k])
            con = ca.vertcat(con, k_con)

        bg = np.zeros(con.shape)
        return con, bg
    def terminal_cost(self, q, target=0, vline_tar=0):
        terminal_cost = (q[0]-target)**2+(q[1]-vline_tar)**2
        terminal_cost *= 0.0
        return terminal_cost
    def initialize(self, x0, K, strategy_cur, npc_states, tr_lights, vline_tar, vline_cur, mode, mat_bnd, ref):
        self.x0 = x0
        self.road_curvature = K # should be 1 X MPC_N array
        self.mat_bnd = mat_bnd
        self.ref = ref

        ## Initial condition
        initial_con = ca.vertcat(self.q[:,0] - self.x0)
        initial_bg = np.zeros(initial_con.shape)

        # Initialize the constraint. allcon
        all_con = initial_con
        self.all_ubg = initial_bg
        self.all_lbg = initial_bg

        try:
            ## model
            model_constraints, model_bg = self.model_equality_constraints()
            all_con = ca.vertcat(all_con, model_constraints)
            self.all_ubg = ca.vertcat(self.all_ubg, model_bg)
            self.all_lbg = ca.vertcat(self.all_lbg, model_bg)
        except:
            print('No model constraint')


        ## env
        self.use_env = use_env.USE_ENV(self.x0, self.MPC_N)
        self.use_env.update(strategy_cur)
        self.use_env.observe(npc_states, tr_lights, vline_tar, vline_cur)
        front_npc, next_npc = self.use_env.grouping_npc()
        # next_npc = self.use_env.
        # try:
        #     terminal_ineq, terminal_hu = self.use_env.construct_terminal_set(mode)
        #
        #
        #     terminal_constraints = terminal_ineq(self.q[:,-1]) - terminal_hu
        #     terminal_ubg = np.zeros(terminal_constraints.shape)
        #     terminal_lbg = np.NINF * np.ones(terminal_ubg.shape)
        #
        #     all_con = ca.vertcat(all_con, terminal_constraints)
        #     self.all_ubg = ca.vertcat(self.all_ubg, terminal_ubg)
        #     self.all_lbg = ca.vertcat(self.all_lbg, terminal_lbg)
        # except:
        #     print('No terminal constraint')
        obj = 0
        try:
            # stage
            stage_ineq_list, stage_hl_list = self.use_env.construct_safety_constr(mode)
            self.slack = ca.MX.sym('ss', 1, len(stage_ineq_list))
            state_con = ca.vertcat(stage_hl_list[0]-stage_ineq_list[0](self.q[:,0]) + 1*self.slack[0,0], self.slack[0,0])
            obj += 0.3*self.slack[0,0]**2
            for k in range(1, len(stage_ineq_list)):
                q_ind = np.mod(k, self.MPC_N+1)
                state_constraints = stage_hl_list[q_ind]-stage_ineq_list[k](self.q[:,q_ind]) + 1*self.slack[0,k]
                state_con = ca.vertcat(state_con, state_constraints, self.slack[0,k])
                obj += 0.3*self.slack[0,k]**2

            state_lbg = np.zeros(state_con.shape) # TODO: should be double-checked
            state_ubg = np.inf * np.ones(state_lbg.shape)

            all_con = ca.vertcat(all_con, state_con)
            self.all_ubg = ca.vertcat(self.all_ubg, state_ubg)
            self.all_lbg = ca.vertcat(self.all_lbg, state_lbg)
            # road bound
            road_ineq_list, road_hl_list = self.use_env.construct_boundary_constr(self.mat_bnd)
            road_con = ca.vertcat(road_hl_list[0]-road_ineq_list[0](self.q[:,0]))
            for k in range(1, len(road_ineq_list)):
                q_ind = np.mod(k, self.MPC_N)
                road_constraints = road_hl_list[q_ind]-road_ineq_list[k](self.q[:,q_ind])
                road_con = ca.vertcat(road_con, road_constraints)
            # road_ubg = np.zeros(road_con.shape) # TODO: should be double-checked
            # road_lbg = np.NINF * np.ones(road_ubg.shape)

            # road_lbg = np.zeros(road_con.shape) # TODO: should be double-checked
            # road_ubg = np.inf * np.ones(road_lbg.shape)
            # all_con = ca.vertcat(all_con, road_con)
            # self.all_ubg = ca.vertcat(self.all_ubg, road_ubg)
            # self.all_lbg = ca.vertcat(self.all_lbg, road_lbg)
        except:
            print('No NPC constraint')

        ## objective
        gamma = 0.1

        ## gain for yaw_desired
        yaw_des_gain = 0.0

        # obj = 0

        if mode == 'LC1':
            if next_npc and abs(next_npc[2]-self.x0[2])>=1.8:
            # if npc_states and abs(npc_states[0][2]-self.x0[2])>=1.8:
                # need to pick the closest vehicle in the next lane                
                npc_state = next_npc
                if self.x0[0] >= npc_state[0]-3:
                    # LC1 : merge in front of that
                    print("mode : LC1")
                    for i in range(self.e.shape[1]):
                        vx_den = self.q[1,i] + 5
                        yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                        obj +=  (ref[0][0,i] - self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 6 * self.u[0,i]**2 + 40*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
                        obj += -0.02*(self.q[0,i]-npc_state[0]-10)**2
                else:
                    # LC2 : merge behind that
                    print("mode: LC2")
                    for i in range(self.e.shape[1]):
                        vx_den = self.q[1,i] + 5
                        yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                        obj +=  2*(npc_state[1]- self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 6 * self.u[0,i]**2 + 40*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
                        obj += -5*(npc_state[0]-self.x0[0])
            else:
                for i in range(self.e.shape[1]):
                    vx_den = self.q[1,i] + 5
                    yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                    obj +=  (ref[0][0,i] - self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 6 * self.u[0,i]**2 + 40*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
        elif mode == 'LK':
            if front_npc:
                # if front_npc exists and close to ego vehicle, we choose v_front as v_ref
                front_s = front_npc[0]
                front_v = front_npc[1]
                front_ey = front_npc[2]
                # close
                if front_s <= self.x0[0] + 20:
                    print("dist mode : {}".format(front_s-self.x0[0]))
                    print("front vel : {}".format(front_v))
                    print("ey gap {}".format(front_ey-self.x0[2]))
                    for i in range(self.e.shape[1]):
                        vx_den = self.q[1,i] + 5
                        yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                        # obj +=  (ref[0][0,i]- self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 6 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
                        obj +=  (front_v- self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 2 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
                # far
                else:
                    for i in range(self.e.shape[1]):
                        vx_den = self.q[1,i] + 5
                        yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                        obj +=  (ref[0][0,i] - self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 2 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
            # no front car
            else:
                for i in range(self.e.shape[1]):
                    vx_den = self.q[1,i] + 5
                    yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                    obj +=  (ref[0][0,i] - self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 2 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
            #Terminal cost
            target = 100
            terminal_cost = self.terminal_cost(self.q[:,-1], target, vline_tar)
            obj += terminal_cost
        elif mode == 'LK_energy':
            if front_npc:
                # if front_npc exists and close to ego vehicle, we choose v_front as v_ref
                front_s = front_npc[0]
                front_v = front_npc[1]
                front_ey = front_npc[2]
                # close
                if front_s <= self.x0[0] + 20:
                    print("dist mode : {}".format(front_s-self.x0[0]))
                    print("front vel : {}".format(front_v))
                    print("ey gap {}".format(front_ey-self.x0[2]))
                    for i in range(self.e.shape[1]):
                        vx_den = self.q[1,i] + 5
                        yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                        # obj +=  (ref[0][0,i]- self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 0*self.e[i] + 6 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
                        obj +=  (front_v- self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 100*self.e[i] + 2 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
                # far
                else:
                    for i in range(self.e.shape[1]):
                        vx_den = self.q[1,i] + 5
                        yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                        obj +=  (ref[0][0,i] - self.q[1,i])**2 + 1*(ref[1]-self.q[2,i])**2 + 100*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 100*self.e[i] + 2 * self.u[0,i]**2 + 10*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
            # no front car
            else:
                for i in range(self.e.shape[1]):
                    vx_den = self.q[1,i] + 5
                    yaw_des = - yaw_des_gain * (self.q[2,i]-ref[1]) / vx_den
                    obj +=  0*(ref[0][0,i] - self.q[1,i])**2 + 0*(ref[1]-self.q[2,i])**2 + 0*(self.q[3,i] - yaw_des)**2 + 0 * (self.q[1,0]*self.u[1,0])**2 + 100*self.e[i] + 0 * self.u[0,i]**2 + 0*(self.u[1,i]-self.road_curvature[0,i]*self.q[1,i]*ca.cos(self.q[3,i]))**2
            #Terminal cost
            target = 100
            terminal_cost = self.terminal_cost(self.q[:,-1], target, vline_tar)
            obj += terminal_cost

        # jerky u minimize
        for i in range(self.e.shape[1]-1):
            obj += 10*(self.u[0,i+1]-self.u[0,i])**2
        ## objective const + vx >=0, (self.q[1,1]**2 - self.q[1,0]**2) (self.q[1,i+1]**2 - self.q[1,i]**2)
        obj_con = ca.vertcat(self.e[0] - (self.q[1,1]**2 - self.q[1,0]**2), self.e[0], self.q[1,0], 10 - self.q[1,0], self.u[0,0]+6, -self.u[0,0]+6, self.q[1,0]*self.u[1,0] + 2.5, -self.q[1,0]*self.u[1,0] + 2.5)
        for i in range(1,self.e.shape[1]):
            obj_con = ca.vertcat(obj_con, self.e[i] - (self.q[1,i+1]**2 - self.q[1,i]**2), self.e[i], self.q[1,i], 10 - self.q[1,i], self.u[0,i]+6, -self.u[0,i]+6, self.q[1,i]*self.u[1,i] + 2.5, -self.q[1,i]*self.u[1,i] + 2.5) #, self.q[1,i]-self.q[1,i-1]+0.2, -self.q[1,i]+self.q[1,i-1]+0.2)

        obj_ubg = np.inf * np.ones(obj_con.shape)
        obj_lbg = np.zeros(obj_con.shape)

        # allcon
        all_con = ca.vertcat(all_con, obj_con)
        self.all_ubg = ca.vertcat(self.all_ubg, obj_ubg)
        self.all_lbg = ca.vertcat(self.all_lbg, obj_lbg)

        try:
            nlp = {'x':ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e), ca.vec(self.slack)), 'f': obj, 'g': all_con}
        except:
            nlp = {'x':ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e)), 'f': obj, 'g': all_con}
        plugin_opts = {'verbose': False, 'verbose_init': False, 'print_time': False, 'ipopt': {"print_level": 0}}
        self.S = ca.nlpsol('S', 'ipopt', nlp, plugin_opts)

    def solve(self):
        try:
            r = self.S(x0=np.zeros(ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e), ca.vec(self.slack)).shape), lbg= self.all_lbg, ubg= self.all_ubg)
        except:
            r = self.S(x0=np.zeros(ca.vertcat(ca.vec(self.q),ca.vec(self.u), ca.vec(self.e)).shape), lbg= self.all_lbg, ubg= self.all_ubg)
        x_opt = r['x']
        # print(r)
        # print(vars(r))

        # state reconstruction
        for i in range(self.MPC_N + 1):
            if i < 1:
                state = np.array(x_opt[0:self.nx])
            else:
                state = np.hstack((state, np.array(x_opt[self.nx * i:self.nx * (i+1)])))


        control_start_ind = self.nx * (self.MPC_N + 1)

        # control input reconstruction
        for i in range(self.MPC_N):
            if i < 1:
                control = np.array(x_opt[control_start_ind:control_start_ind+self.nu])
            else:
                control = np.hstack((control, np.array(x_opt[control_start_ind + self.nu * i:control_start_ind + self.nu * (i+1)])))


        energy_start_ind = self.nx * (self.MPC_N + 1) + self.nu * (self.MPC_N)
        # energy reconstruction
        for i in range(self.MPC_N):
            if i < 1:
                energy = np.array(x_opt[energy_start_ind:energy_start_ind+1])
            else:
                energy = np.hstack((energy, np.array(x_opt[energy_start_ind + i:energy_start_ind + (i+1)])))

        return state, control, energy # np.array type. nx x mpc_N + 1, nu x mpc_N, 1 x mpc_N
