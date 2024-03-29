import carla
import os
import sys
import numpy as np
import casadi
import time

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

sys.path.append(CARLA_ROOT + "PythonAPI/carla/agents/")
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from utils import frenet_trajectory_handler as fth
from utils.low_level_control import LowLevelControl
from utils.vehicle_geometry_utils import vehicle_name_to_lf_lr
import matplotlib.pyplot as plt

class MPCAgent(object):
    """ A path following agent with collision avoidance constraints over a short horizon. """

    def __init__(self,
                 vehicle,
                 goal_location,
                 nominal_speed_mps =8.0, # sets desired speed (m/s) for tracking path
                 dt =0.2,
                 N=8,                   # time discretization (s) used to generate a reference
                 N_modes = 3):
        self.vehicle = vehicle
        self.world   = vehicle.get_world()
        carla_map     = self.world.get_map()
        planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(carla_map, sampling_resolution=0.5) )
        planner.setup()

        # Get the high-level route using Carla's API (basically A* search over road segments).
        init_waypoint = carla_map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        goal          = carla_map.get_waypoint(goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
        route = planner.trace_route(init_waypoint.transform.location, goal.transform.location)

        # Convert the high-level route into a path parametrized by arclength distance s (i.e. Frenet frame).
        way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
        self._frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)

        # TODO: remove hard-coded values.
        self.nominal_speed = nominal_speed_mps # m/s
        self.lat_accel_max = 2.0  # m/s^2

        self.lf, self.lr = vehicle_name_to_lf_lr(self.vehicle.type_id)
        self._setup_mpc(N=N, DT=dt, L_F=self.lf, L_R=self.lr)


        self._fit_velocity_profile()

        self._low_level_control = LowLevelControl(vehicle)
        self.goal_reached = False # flags when the end of the path is reached and agent should stop
        self.counter = 0

    def done(self):
        return self.goal_reached

    def run_step(self, pred_dict):
        vehicle_loc   = self.vehicle.get_location()
        vehicle_tf    = self.vehicle.get_transform()
        vehicle_vel   = self.vehicle.get_velocity()
        vehicle_accel = self.vehicle.get_acceleration()
        speed_limit   = self.nominal_speed #self.vehicle.get_speed_limit()

        # Get the vehicle's current pose in a RH coordinate system.
        x, y = vehicle_loc.x, -vehicle_loc.y
        psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))

        # Get the current speed and longitudinal acceleration.
        speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)  # TODO: fix?
        accel = np.cos(psi) * vehicle_accel.x - np.sin(psi)*vehicle_accel.y

        # Look up the projection of the current pose to Frenet frame.
        s, ey, epsi = \
            self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
        curv = self._frenet_traj.get_curvature_at_s(s)

        # Initialize variables to be returned.
        z0=np.array([x,y,psi,speed])
        u0=np.array([self.A_MIN, 0.])
        v_des = np.clip(z0[-1] + self.A_MIN * self.DT, self.V_MIN, self.V_MAX)
        is_opt=False
        solve_time=np.nan

        if self.goal_reached or self._frenet_traj.reached_trajectory_end(s, resolution=5.):
            # Stop if the end of the path is reached and signal completion.
            self.goal_reached = True
        else:
            # Update MPC problem.
            update_dict = {'x0'      : x,
                           'y0'      : y,
                           'psi0'    : psi,
                           'v0'      : speed}
            update_dict.update( self._get_reference_traj(**update_dict) )
            update_dict['tv_refs'], update_dict['tv_Rs']  = self._get_target_vehicles(x, y)

            if self.warm_start:
                update_dict['acc_prev']   = self.warm_start['u_ws'][0, 0]
                update_dict['df_prev']    = self.warm_start['u_ws'][0, 1]
                update_dict['warm_start'] = self.warm_start
            else:
                update_dict['acc_prev']  = 0.
                update_dict['df_prev']   = 0.

            self._update(update_dict)

            # Solve MPC problem.
            sol_dict = self._solve()
            if sol_dict['optimal']:
                self.warm_start = {}
                self.warm_start['z_ws']       = sol_dict['z_mpc']
                self.warm_start['u_ws']       = sol_dict['u_mpc']
                self.warm_start['sl_ws']      = sol_dict['sl_mpc']

            u0=sol_dict['u_mpc'][0,:]
            is_opt     = sol_dict['optimal']
            solve_time = sol_dict['solve_time']
            v_des = sol_dict['z_mpc'][1,3]

        # Get low level control.
        control =  self._low_level_control.update(speed, # v_curr
                                                  u0[0], # a_des
                                                  v_des, # v_des
                                                  u0[1]) # df_des

        # if self.counter % 10 == 0:
        #     plt.subplot(311)
        #     plt.plot(-sol_dict['z_ref'][:,1], sol_dict['z_ref'][:,0], 'kx')
        #     plt.plot(-sol_dict['z_mpc'][:,1], sol_dict['z_mpc'][:,0], 'r')

        #     for tv_ref in sol_dict['tv_refs']:
        #         plt.plot(-tv_ref[:,1], tv_ref[:,0], 'b')
        #     plt.ylim([0,50]); plt.xlim([-150, 150])

        #     plt.xlabel('y'); plt.ylabel('x')

        #     plt.subplot(312)
        #     plt.plot(np.array([x*self.DT for x in range(1, self.N+1)]), sol_dict['z_ref'][:, 2], 'kx')
        #     plt.plot(np.array([x*self.DT for x in range(1, self.N+1)]), sol_dict['z_mpc'][1:, 2], 'r')

        #     plt.subplot(313)
        #     plt.plot(np.array([x*self.DT for x in range(1, self.N+1)]), sol_dict['z_ref'][:, 3], 'kx')
        #     plt.plot(np.array([x*self.DT for x in range(1, self.N+1)]), sol_dict['z_mpc'][1:, 3], 'r')

        #     plt.suptitle(f"ACC:{sol_dict['u_mpc'][0,0]}, V:{sol_dict['z_mpc'][1,3]}, ST:{sol_dict['u_mpc'][0,1]}")

        #     plt.show()


        return control, z0, u0, is_opt, solve_time

    ################################################################################################
    ########################## Helper / Update Functions ###########################################
    ################################################################################################
    def _fit_velocity_profile(self):

        t_fits = [0.]
        traj = self._frenet_traj.trajectory

        for state, next_state in zip(traj[:-1, :], traj[1:, :]):
            s, x, y, yaw, curv = state
            sn, xn, yn, yawn, curvn = next_state

            v_curr = min( self.nominal_speed, np.sqrt(self.lat_accel_max / max(0.01, np.abs(curv))) )

            t_fits.append( (sn - s) / v_curr + t_fits[-1] )

        # Interpolate the points at time discretization dt.
        t_disc    = np.arange(t_fits[0], t_fits[-1] + self.DT/2, self.DT)
        s_disc    = np.interp(t_disc, t_fits, traj[:,0])
        x_disc    = np.interp(t_disc, t_fits, traj[:,1])
        y_disc    = np.interp(t_disc, t_fits, traj[:,2])
        yaw_disc  = np.interp(t_disc, t_fits, traj[:,3])
        # curv_disc = np.interp(t_disc, t_fits, traj[:,4])

        v_disc    = np.diff(s_disc) / np.diff(t_disc)
        v_disc    = np.insert(v_disc, -1, v_disc[-1]) # repeat the last speed

        self.reference = np.column_stack((t_disc, x_disc, y_disc, yaw_disc, v_disc))

    def _get_reference_traj(self, x0, y0, psi0, v0):
        ref_dict = {}

        closest_idx = np.argmin( np.linalg.norm(self.reference[:, 1:3] - np.array([x0, y0]), axis=-1) )

        t_ref = self.reference[closest_idx, 0] + np.array([x*self.DT for x in range(1, self.N+1)])

        ref_dict['x_ref']   = np.interp(t_ref, self.reference[:, 0], self.reference[:, 1])
        ref_dict['y_ref']   = np.interp(t_ref, self.reference[:, 0], self.reference[:, 2])
        ref_dict['v_ref']   = np.interp(t_ref, self.reference[:, 0], self.reference[:, 4])

        ref_dict['psi_ref'] = np.interp(t_ref, self.reference[:, 0], self.reference[:, 3])
        ref_dict['psi_ref'] = fth.fix_angle( ref_dict['psi_ref'] - psi0) + psi0

        return ref_dict

    def _get_target_vehicles(self, ego_x, ego_y):
        all_actors = self.world.get_actors()
        veh_actors = all_actors.filter('vehicle*')
        ped_actors = all_actors.filter('walker*')

        def get_drel(actor):
            act_loc = actor.get_location()
            act_x, act_y = act_loc.x, -act_loc.y

            return ((act_x - ego_x)**2 + (act_y - ego_y)**2)**0.5

        act_id_drel_map = {} # TODO: consider relative velocity.

        for veh_act in veh_actors:
            if self.vehicle.id == veh_act.id:
                continue
            act_id_drel_map[veh_act.id] = get_drel(veh_act)

        for ped_act in ped_actors:
            act_id_drel_map[ped_act.id] = get_drel(ped_act)

        # Filter by proximity (TODO: improve this).
        act_ids_drel_ordered = sorted(act_id_drel_map.items(), key=lambda x:x[1])

        def get_short_term_pred_tv(actor, N):
            act_loc    = actor.get_location()
            act_vel    = actor.get_velocity()
            act_angvel = actor.get_angular_velocity()
            act_tf     = actor.get_transform()

            act_x, act_y =  act_loc.x, -act_loc.y
            act_psi      = -fth.fix_angle(np.radians(act_tf.rotation.yaw))
            act_v        =  act_vel.x * np.cos(act_psi) - act_vel.y * np.sin(act_psi)
            act_w        = -np.radians(act_angvel.z)

            x_preds = []
            y_preds = []
            R_preds = []

            for _ in range(self.N_PRED_TV):
                act_xn = act_x   + act_v * np.cos(act_psi) * self.DT
                act_yn = act_y   + act_v * np.sin(act_psi) * self.DT
                act_pn = act_psi + act_w * self.DT

                x_preds.append(act_xn)
                y_preds.append(act_yn)
                R_preds.append(np.array([[np.cos(act_pn), np.sin(act_pn)],[-np.sin(act_pn), np.cos(act_pn)]]))

                act_x   = act_xn
                act_y   = act_yn
                act_psi = act_pn

            return np.column_stack((x_preds, y_preds)), R_preds

        tv_refs = []
        Rs =[]
        for idx in range(self.NUM_TVS):
            if idx < len(act_ids_drel_ordered):
                actor = all_actors.find( act_ids_drel_ordered[idx][0] )
                tv_ref, R = get_short_term_pred_tv(actor, self.N_PRED_TV)
                tv_refs.append( tv_ref )
                Rs.append( R)
            else:
                fake_agent_position = [ego_x + 1000., ego_y + 1000.]
                tv_ref = np.array(self.N_PRED_TV*[fake_agent_position])
                Rs.append([ np.identity(2) ]*self.N_PRED_TV)
                tv_refs.append( tv_ref )
        return tv_refs, Rs

    ################################################################################################
    ################################ MPC Formulation ###############################################
    ################################################################################################
    def _setup_mpc(self,
                   N          =   10,   # timesteps in MPC Horizon
                   DT         =  0.2,   # discretization time between timesteps (s)
                   N_PRED_TV  =    3,   # timesteps for target vehicle prediction
                   NUM_TVS    =    5,   # maximum number of target vehicles to avoid
                   D_MIN      =  0.5,  # square of minimum 2-norm distance to a target vehicle
                   L_F        =  1.5,   # distance from CoG to front axle (m) [guesstimate]
                   L_R        =  1.5,   # distance from CoG to rear axle (m) [guesstimate]
                   V_MIN      =  0.0,   # min/max velocity constraint (m/s)
                   V_MAX      = 20.0,
                   A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
                   A_MAX      =  2.0,
                   DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
                   DF_MAX     =  0.5,
                   A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
                   A_DOT_MAX  =  1.5,
                   DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
                   DF_DOT_MAX =  0.5,
                   C_OBS_SL   = 1000,      # weights for slack on collision avoidance (norm constraint).
                   Q = [1., 1., 50., 0.1], # weights on x, y, and v.
                   R = [100., 1000.],
                   fps=20):
                   # Q = [1., 1., 10., 0.1], # weights on x, y, psi, and v.
                   # R = [10., 100.]):       # weights on jerk and slew rate (steering angle derivative)

        for key in list(locals()):
            if key == 'self':
                pass
            elif key == 'Q':
                self.Q = casadi.diag(Q)
            elif key == 'R':
                self.R = casadi.diag(R)
            else:
                setattr(self, '%s' % key, locals()[key])

        self.opti = casadi.Opti()

        """ Parameters """
        self.u_prev  = self.opti.parameter(2)       # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_curr  = self.opti.parameter(4)       # current state:  [x_0, y_0, psi_0, v_0]

        self.z_ref = self.opti.parameter(self.N, 4) # reference trajectory starting at timestep 1.
        self.x_ref   = self.z_ref[:,0]
        self.y_ref   = self.z_ref[:,1]
        self.psi_ref = self.z_ref[:,2]
        self.v_ref   = self.z_ref[:,3]

        # Collision Avoidance: center positions for each target vehicle (spoofed if not present).
        self.tv_refs = [ self.opti.parameter(self.N_PRED_TV, 2) for _ in range(self.NUM_TVS) ]
        self.tv_Rs   = [ [self.opti.parameter(2,2) for _ in range(self.N_PRED_TV)] for _ in range(self.NUM_TVS) ]
        """ Decision Variables """
        self.z_dv = self.opti.variable(self.N+1, 4)  # solution trajectory starting at timestep 0.
        self.x_dv   = self.z_dv[:, 0]
        self.y_dv   = self.z_dv[:, 1]
        self.psi_dv = self.z_dv[:, 2]
        self.v_dv   = self.z_dv[:, 3]

        self.u_dv = self.opti.variable(self.N, 2)    # input trajectory starting at timestep 0
        self.acc_dv = self.u_dv[:,0]
        self.df_dv  = self.u_dv[:,1]

        self.sl_dv  = self.opti.variable(self.N , 2) # slack variables for input rate constraint.
        self.sl_acc_dv = self.sl_dv[:,0]
        self.sl_df_dv  = self.sl_dv[:,1]

        self.sl_obst_dv = self.opti.variable(self.N_PRED_TV) # slack variable for obstacle avoidance
        self.S=casadi.DM([[(3+self.D_MIN)**(-2), 0.],[0., (1+self.D_MIN)**(-2)]])

        self._add_agent_constraints()
        self._add_obstacle_avoidance_constraints()

        self._add_cost()

        self._update_initial_condition(0., 0., 0., 1.)

        self._update_reference([self.DT * (x+1) for x in range(self.N)],
                               self.N*[0.],
                               self.N*[0.],
                               self.N*[1.])

        self._update_previous_input(0., 0.)

        tv_refs_fake = [ 1000. * np.ones((self.N_PRED_TV,2)) for _ in range(self.NUM_TVS) ]
        tv_Rs_fake   =  [[np.identity(2)]*self.N_PRED_TV]*self.NUM_TVS
        self._update_obstacles(tv_refs_fake, tv_Rs_fake)

        self.opti.solver("ipopt", {"expand": False}, {"max_cpu_time": 0.1, "print_level": 0})

        sol = self._solve()

        self.warm_start = None

    def _add_agent_constraints(self):
        # State Bound Constraints
        self.opti.subject_to( self.opti.bounded(self.V_MIN, self.v_dv, self.V_MAX) )

        # Initial State Constraint
        self.opti.subject_to( self.x_dv[0]   == self.z_curr[0] )
        self.opti.subject_to( self.y_dv[0]   == self.z_curr[1] )
        self.opti.subject_to( self.psi_dv[0] == self.z_curr[2] )
        self.opti.subject_to( self.v_dv[0]   == self.z_curr[3] )

        # State Dynamics Constraints
        for i in range(self.N):
            beta = casadi.atan( self.L_R / (self.L_F + self.L_R) * casadi.tan(self.df_dv[i]) )
            self.opti.subject_to( self.x_dv[i+1]   == self.x_dv[i]   + self.DT * (self.v_dv[i] * casadi.cos(self.psi_dv[i] + beta)) )
            self.opti.subject_to( self.y_dv[i+1]   == self.y_dv[i]   + self.DT * (self.v_dv[i] * casadi.sin(self.psi_dv[i] + beta)) )
            self.opti.subject_to( self.psi_dv[i+1] == self.psi_dv[i] + self.DT * (self.v_dv[i] / self.L_R * casadi.sin(beta)) )
            self.opti.subject_to( self.v_dv[i+1]   == self.v_dv[i]   + self.DT * (self.acc_dv[i]) )

        # Input Bound Constraints
        self.opti.subject_to( self.opti.bounded(self.A_MIN,  self.acc_dv, self.A_MAX) )
        self.opti.subject_to( self.opti.bounded(self.DF_MIN, self.df_dv,  self.DF_MAX) )

        # Input Rate Bound Constraints
        self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.fps**(-1) -  self.sl_acc_dv[0],
                                                 self.acc_dv[0] - self.u_prev[0],
                                                 self.A_DOT_MAX*self.fps**(-1)   + self.sl_acc_dv[0]) )

        self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.fps**(-1)  -  self.sl_df_dv[0],
                                                 self.df_dv[0] - self.u_prev[1],
                                                 self.DF_DOT_MAX*self.fps**(-1)  + self.sl_df_dv[0]) )

        for i in range(self.N - 1):
            self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT   -  self.sl_acc_dv[i+1],
                                                     self.acc_dv[i+1] - self.acc_dv[i],
                                                     self.A_DOT_MAX*self.DT   + self.sl_acc_dv[i+1]) )
            self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[i+1],
                                                     self.df_dv[i+1]  - self.df_dv[i],
                                                     self.DF_DOT_MAX*self.DT  + self.sl_df_dv[i+1]) )
        # Slack Constraints
        self.opti.subject_to( 0 <= self.sl_df_dv )
        self.opti.subject_to( 0 <= self.sl_acc_dv )

    @staticmethod
    def _quad_form(z, Q):
        return casadi.mtimes(z, casadi.mtimes(Q, z.T))

    def _add_obstacle_avoidance_constraints(self):
        self.opti.subject_to( 0== self.sl_obst_dv )

        for k, tv_ref in enumerate(self.tv_refs):
            for i in range(self.N_PRED_TV):
                self.opti.subject_to( 1.0- self.sl_obst_dv[i] <= self._quad_form(self.z_dv[i+1, :2] - tv_ref[i, :], self.tv_Rs[k][i].T@self.S@self.tv_Rs[k][i] ))

    def _add_cost(self):
        cost = 0
        for i in range(self.N):
            cost += self._quad_form(self.z_dv[i+1, :] - self.z_ref[i,:], self.Q) # tracking cost

        for i in range(self.N - 1):
            cost += self._quad_form(self.u_dv[i+1, :] - self.u_dv[i,:], self.R)  # input derivative cost

        cost += (casadi.sum1(self.sl_df_dv) + casadi.sum1(self.sl_acc_dv))  # slack cost

        cost += self.C_OBS_SL * casadi.sum1(self.sl_obst_dv)

        self.opti.minimize( cost )

    def _update_initial_condition(self, x0, y0, psi0, vel0):
        self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0])

    def _update_reference(self, x_ref, y_ref, psi_ref, v_ref):
        self.opti.set_value(self.x_ref,   x_ref)
        self.opti.set_value(self.y_ref,   y_ref)
        self.opti.set_value(self.psi_ref, psi_ref)
        self.opti.set_value(self.v_ref,   v_ref)

    def _update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])

    def _update_obstacles(self, tv_refs, tv_Rs):
        assert len(tv_refs) == self.NUM_TVS
        for i, tv_ref in enumerate(tv_refs):
            self.opti.set_value(self.tv_refs[i], tv_ref)
        for k in range(self.NUM_TVS):
            for j in range(self.N_PRED_TV):
                self.opti.set_value(self.tv_Rs[k][j], tv_Rs[k][j])

    def _update(self, update_dict):
        self._update_initial_condition( *[update_dict[key] for key in ['x0', 'y0', 'psi0', 'v0']] )
        self._update_reference( *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref']] )
        self._update_previous_input( *[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self._update_obstacles( update_dict['tv_refs'], update_dict['tv_Rs'] )

        if 'warm_start' in update_dict.keys():
            # Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
            self.opti.set_initial(self.z_dv,  update_dict['warm_start']['z_ws'])
            self.opti.set_initial(self.u_dv,  update_dict['warm_start']['u_ws'])
            self.opti.set_initial(self.sl_dv, update_dict['warm_start']['sl_ws'])

    def _solve(self):
        st = time.time()
        try:
            sol = self.opti.solve()
            # Optimal solution.
            u_mpc  = sol.value(self.u_dv)
            z_mpc  = sol.value(self.z_dv)
            sl_mpc = sol.value(self.sl_dv)
            sl_obst_mpc = sol.value(self.sl_obst_dv)
            z_ref  = sol.value(self.z_ref)
            tv_refs = [ sol.value(self.tv_refs[i]) for i in range(self.NUM_TVS) ]
            is_opt = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_mpc  = self.opti.debug.value(self.u_dv)
            z_mpc  = self.opti.debug.value(self.z_dv)
            sl_mpc = self.opti.debug.value(self.sl_dv)
            sl_obst_mpc = self.opti.debug.value(self.sl_obst_dv)
            z_ref  = self.opti.debug.value(self.z_ref)
            tv_refs = [ self.opti.debug.value(self.tv_refs[i]) for i in range(self.NUM_TVS) ]
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']   = u_mpc[0,:]  # control input to apply based on solution
        sol_dict['optimal']     = is_opt      # whether the solution is optimal or not
        sol_dict['solve_time']  = solve_time  # how long the solver took in seconds
        sol_dict['u_mpc']       = u_mpc       # solution inputs (N by 2, see self.u_dv above)
        sol_dict['z_mpc']       = z_mpc       # solution states (N+1 by 4, see self.z_dv above)
        sol_dict['sl_mpc']      = sl_mpc      # solution slack vars (N by 2, see self.sl_dv above)
        sol_dict['sl_obst_mpc'] = sl_obst_mpc # solution slack vars for collision avoidance
        sol_dict['z_ref']       = z_ref       # state reference (N by 4, see self.z_ref above)
        sol_dict['tv_refs']     = tv_refs     # reference for target vehicles (to avoid)

        return sol_dict
