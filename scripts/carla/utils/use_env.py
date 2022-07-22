'''
© 2021 Hotae Lee <hotae.lee@berkeley.edu>
Observe environments and convert into constraints/sets for MPC
'''
# Import general packages.
import numpy as np
import matplotlib.pyplot as plt
import time
# Import optimization solvers
import casadi as ca

class USE_ENV: #USE_ENV(EGO_VEHICLE)
    def __init__(self, ego_state, N_MPC, dt =0.1):
        self.ego_state = ego_state
        self.N_MPC = N_MPC
        self.dt = dt
        self.safe_time_mpc = 2
        self.safe_time_planner = 1
        self.safety_dist = ego_state[1]*self.safe_time_mpc + 14# sdot*time
        self.safety_dist_planner = ego_state[1]*self.safe_time_planner + 22# sdot*time
    def update(self, strategy):
        self.strategy = strategy # [pass, pass, stop] for next several traffic lights
    def observe(self, npc_states, tr_lights, vline_tar, vline_cur):
        # Surronding Vehicles
        self.npc_states = npc_states # list of x = [s,sdot,y,psi] / u = [acc,steering]
        # TO DISCUSS : If the data comes to us in the form of distance from ego, we can use it directly
        self.Nnpc = len(npc_states)
        self.vline_tar = vline_tar # target lane
        self.vline_cur = vline_cur # current lane
        # Traffic lights
        self.light_dist = tr_lights[0].dist
        self.light_phase = tr_lights[0].phase
        self.light_timing = tr_lights[0].timing # remaining time to change (T_traffic - T_cur)
        # Lane (Curvatur and width)
        self.lane_width = 2.2
    def grouping_npc(self):
        # TODO : need a function to split into two groups (same lane & other lane) and sort
        self.npcs_otherlane = []
        self.npcs_samelane = []
        for npc_state in self.npc_states:
            print("ego y={},npc y{}".format(self.ego_state[2], npc_state[2]))
            if abs(npc_state[2]-self.ego_state[2]) <= self.lane_width/2:
                # regard it as the same lane
                self.npcs_samelane.append(npc_state)
            else:
                # regard it as the other lane
                self.npcs_otherlane.append(npc_state)

        print("same: {}, other : {}".format(len(self.npcs_samelane), len(self.npcs_otherlane)))
        gapmin = 100
        front_npc = []
        next_npc = []
        next_front_npc = []
        next_rear_npc = []

        # Find a front npc
        for npc_state in self.npcs_samelane:
            if (self.ego_state[0] <= npc_state[0] and npc_state[0] - self.ego_state[0] <=gapmin):
                gapmin = npc_state[0]-self.ego_state[0]
                front_npc = npc_state
        # Find a next npc (Closest s)
        gapmin = 50
        for npc_state in self.npcs_otherlane:
            if npc_state[0] - self.ego_state[0] <=gapmin:
                gapmin = npc_state[0]-self.ego_state[0]
                next_npc = npc_state
        print("next npc debug", next_npc, self.npcs_otherlane, "ego state", self.ego_state)
        # find a front car of next car
        gapmin = 50
        if next_npc:
            for npc_state in self.npcs_otherlane:
                if abs(npc_state[0] - next_npc[0]) <=gapmin and next_npc[0] <npc_state[0]:
                    gapmin = npc_state[0]-next_npc[0]
                    next_front_npc = npc_state
            # find a rear car of next car
            gapmin = 50
            for npc_state in self.npcs_otherlane:
                if abs(npc_state[0] - next_npc[0]) <=gapmin and next_npc[0] > npc_state[0]:
                    gapmin = npc_state[0]-next_npc[0]
                    next_rear_npc = npc_state
        next_npcs = (next_front_npc, next_npc, next_rear_npc)
        return front_npc, next_npcs
    def construct_terminal_set(self, mode):
        # forcespro ex) model.ineq[-1] = terminal_ineq, model.hu[-1] = terminal_hu

        if self.strategy[0] == 1:
            # if pass
            if mode in ['LC1','LC2']:
                s_terminal = self.light_dist - self.vline_tar*(self.light_timing-self.dt*self.N_MPC)
            elif mode == 'LK':
                s_terminal = self.light_dist - self.vline_cur*(self.light_timing-self.dt*self.N_MPC)
            # return the parameters in the form of g(x) <=0
            terminal_ineq = lambda x : s_terminal - x[0]
            terminal_hu = 0
        else:
            # if stop
            s_terminal = self.light_dist
            # return the parameters in the form of g(x) <=0
            terminal_ineq = lambda x :  x[0] - s_terminal
            terminal_hu = 0
        # if we want to increase # of ineq, ca.vertcat or np.vstack & np.append
        return terminal_ineq, terminal_hu
    def construct_safety_constr(self, mode):
        # forcespro ex) model.ineq[k] = stage_ineq_list[k], model.hl[k] = stage_hl_list[k]
        # casadi nlp ex) nlp('x', [x;y], 'f', x**2, 'g', terminal_ineq[k](x[k]))
        stage_ineq_list = []
        stage_hl_list = []
        if mode == 'LC1' or mode == 'LC':
            for npc_state in self.npcs_otherlane:
                for k in range(self.N_MPC):
                    # Prediction with a simple model (const. vel + no lane change)
                    npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                    stage_ineq = lambda x : -((x[0]-npc_s_pred)**2 + (x[2]-npc_state[2])**2)
                    stage_hl = -self.lane_width
                    stage_ineq_list.append(stage_ineq)
                    stage_hl_list.append(stage_hl)
            for npc_state in self.npcs_samelane:
                # if npc_state[0] >= self.ego_state[0]:
                for k in range(self.N_MPC):
                    npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                    stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                    stage_hl = -(self.safety_dist-npc_state[1]*self.safe_time_mpc)**2
                    stage_ineq_list.append(stage_ineq)
                    stage_hl_list.append(stage_hl)
        elif mode == 'LC2':
            npc_state = self.npcs_otherlane[0] # suppose the frontmost
            for k in range(self.N_MPC):
                # Prediction with a simple model (const. vel + no lane change)
                npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                stage_hl = -(self.safety_dist-npc_state[1]*self.safe_time_mpc)**2
                stage_ineq_list.append(stage_ineq)
                stage_hl_list.append(stage_hl)
            npc_state = self.npcs_otherlane[1] # suppose the second
            for k in range(self.N_MPC):
                # Prediction with a simple model (const. vel + no lane change)
                npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                stage_hl = -(self.safety_dist-npc_state[1]*self.safe_time_mpc)**2
                stage_ineq_list.append(stage_ineq)
                stage_hl_list.append(stage_hl)
            for npc_state in self.npcs_samelane:
                if npc_state[0] >= self.ego_state[0]:
                    for k in range(self.N_MPC):
                        npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                        stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                        stage_hl = -(self.safety_dist-npc_state[1]*self.safe_time_mpc)**2
                        stage_ineq_list.append(stage_ineq)
                        stage_hl_list.append(stage_hl)
        else:
            for npc_state in self.npcs_samelane:
                if npc_state[0] >= self.ego_state[0]:
                    for k in range(self.N_MPC+1):
                        npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                        stage_ineq = lambda x : x[0]-npc_s_pred
                        stage_hl = -(self.safety_dist-npc_state[1]*self.safe_time_mpc)
                        stage_ineq_list.append(stage_ineq)
                        stage_hl_list.append(stage_hl)
        return stage_ineq_list, stage_hl_list
    def construct_safety_constr_planner(self, mode):
        # forcespro ex) model.ineq[k] = stage_ineq_list[k], model.hl[k] = stage_hl_list[k]
        # casadi nlp ex) nlp('x', [x;y], 'f', x**2, 'g', terminal_ineq[k](x[k]))
        stage_ineq_list = []
        stage_hl_list = []
        if mode == 'LC1':
            for npc_state in self.npcs_otherlane:
                for k in range(self.N_MPC):
                    # Prediction with a simple model (const. vel + no lane change)
                    npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                    stage_ineq = lambda x : -((x[0]-npc_s_pred)**2 + (x[2]-npc_state[2])**2)
                    stage_hl = -self.lane_width
                    stage_ineq_list.append(stage_ineq)
                    stage_hl_list.append(stage_hl)
            for npc_state in self.npcs_samelane:
                # if npc_state[0] >= self.ego_state[0]:
                for k in range(self.N_MPC):
                    npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                    stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                    stage_hl = -(self.safety_dist_planner-npc_state[1]*self.safe_time_planner)**2
                    stage_ineq_list.append(stage_ineq)
                    stage_hl_list.append(stage_hl)
        elif mode == 'LC2':
            npc_state = self.npcs_otherlane[0] # suppose the frontmost
            for k in range(self.N_MPC):
                # Prediction with a simple model (const. vel + no lane change)
                npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                stage_hl = -(self.safety_dist_planner-npc_state[1]*self.safe_time_planner)**2
                stage_ineq_list.append(stage_ineq)
                stage_hl_list.append(stage_hl)
            npc_state = self.npcs_otherlane[1] # suppose the second
            for k in range(self.N_MPC):
                # Prediction with a simple model (const. vel + no lane change)
                npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                stage_hl = -(self.safety_dist_planner-npc_state[1]*self.safe_time_planner)**2
                stage_ineq_list.append(stage_ineq)
                stage_hl_list.append(stage_hl)
            for npc_state in self.npcs_samelane:
                if npc_state[0] >= self.ego_state[0]:
                    for k in range(self.N_MPC):
                        npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                        stage_ineq = lambda x : -(x[0]-npc_s_pred)**2
                        stage_hl = -(self.safety_dist_planner-npc_state[1]*self.safe_time_planner)**2
                        stage_ineq_list.append(stage_ineq)
                        stage_hl_list.append(stage_hl)
        else:
            for npc_state in self.npcs_samelane:
                if npc_state[0] >= self.ego_state[0]:
                    for k in range(self.N_MPC+1):
                        npc_s_pred = npc_state[0] + npc_state[1]*k*self.dt
                        stage_ineq = lambda x : -(npc_s_pred-x[0])
                        stage_hl = -(self.safety_dist_planner-npc_state[1]*self.safe_time_planner)
                        stage_ineq_list.append(stage_ineq)
                        stage_hl_list.append(stage_hl)
        return stage_ineq_list, stage_hl_list
    def construct_boundary_constr(self, mat_bnd):
        # road_bnd1, road_bnd2
        bnd1 = mat_bnd['road_bnd1']
        bnd2 = mat_bnd['road_bnd2']
        road_ineq_list = []
        road_hl_list = []
        for k in range(self.N_MPC):
            road_ineq1 = lambda x : -x[2]
            road_hl1 = -bnd1
            road_ineq2 = lambda x : x[2]
            road_hl2 = bnd2
            road_ineq_list.append(road_ineq1)
            road_hl_list.append(road_hl1)
            road_ineq_list.append(road_ineq2)
            road_hl_list.append(road_hl2)
        return road_ineq_list, road_hl_list

# To check the functionality and explain how to use
class Ego:
    def __init__(self, s, sdot, ey, psi):
        self.state = [s,sdot,ey,psi]
class tr_light:
    def __init__(self, dist, phase, timing):
        self.dist = dist
        self.phase = phase
        self.timing = timing

if __name__ == "__main__":
    lane_width = 2
    # Define ego, use_env
    ego = Ego(0,0,0,0)
    N_MPC = 3
    use_env = USE_ENV(ego.state,N_MPC)
    # recieve strategy and observe
    strategy_cur = [0,0]
    npc_state1 = [5,0,0,0]
    npc_state2 = [5,0,-lane_width,0]
    npc_states = [npc_state1, npc_state2]
    print(npc_states)
    tr1 = tr_light(20,'red',5)
    tr2 = tr_light(50,'red',5)
    tr_lights = [tr1, tr2]
    vline_tar = 5
    vline_cur = 2
    #
    use_env.update(strategy_cur)
    use_env.observe(npc_states, tr_lights, vline_tar, vline_cur)
    use_env.grouping_npc()
    # construct constraints
    t_cur = 1
    N_MPC = 3
    mode_list = ['LC1','LC2','LK']
    mode = mode_list[0]
    terminal_ineq, terminal_hu = use_env.construct_terminal_set(mode)
    stage_ineq_list, stage_hl_list = use_env.construct_safety_constr(mode)
    print(stage_ineq_list)
