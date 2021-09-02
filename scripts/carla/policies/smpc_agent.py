import carla
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.signal import filtfilt
CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

sys.path.append(CARLA_ROOT + "PythonAPI/carla/agents/")
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from evaluation.gmm_prediction import GMMPrediction

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from utils import frenet_trajectory_handler as fth
from utils import mpc_utils as smpc
from utils.low_level_control import LowLevelControl
from utils.vehicle_geometry_utils import vehicle_name_to_lf_lr

class SMPCAgent(object):
    """ Implementation of an agent using multimodal predictions and stochastic MPC for control. """

    def __init__(self,
                 vehicle,                  # Vehicle object that this agent controls
                 goal_location,            # desired goal location used to generate a path
                 nominal_speed_mps =8.0, # sets desired speed (m/s) for tracking path
                 dt =0.2,
                 N=8,                   # time discretization (s) used to generate a reference
                 N_modes = 3,
                 smpc_config = "full"
                 ):
        self.vehicle = vehicle
        self.map    = vehicle.get_world().get_map()
        self.dt      = dt
        self.goal_location = goal_location
        self.nominal_speed_mps  = nominal_speed_mps
        self.N=N
        self.N_modes=N_modes
        self.planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(self.map, sampling_resolution=0.5) )
        self.planner.setup()
        self.lf, self.lr = vehicle_name_to_lf_lr(self.vehicle.type_id)
        self._low_level_control = LowLevelControl(vehicle)
        self.time=0
        self.t_ref=0
        if smpc_config=="full":
            self.ol_flag=False
            self.ns_bl_flag=False
        elif smpc_config=="open_loop":
            self.ol_flag=True
            self.ns_bl_flag=False
        elif smpc_config=="no_switch":
            self.ol_flag=False
            self.ns_bl_flag=True
        else:
            raise ValueError(f"Invalid SMPC config: {smpc_config}")

        # Get the high-level route using Carla's API (basically A* search over road segments).
        # init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        # goal          = self.map.get_waypoint(goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
        # route = self.planner.trace_route(init_waypoint.transform.location, goal.transform.location)

        # # Convert the high-level route into a path parametrized by arclength distance s (i.e. Frenet frame).
        # # Generate a refernece by fitting a velocity profile with specified nominal speed and time discretization.
        # way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
        # self.frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)
        # self.nominal_speed = nominal_speed_mps
        # self.lat_accel_max = 3.0 # maximum lateral acceleration (m/s^2), for slowing down at turns
        # self.dt = dt
        # self.fit_velocity_profile()

        # # Feasible reference generation by nonlinear trajectory optimization
        # self.feas_ref_gen=smpc.RefTrajGenerator(N=self.ref_horizon, DT=dt)
        # self.feas_ref_gen.update(self.ref_dict)
        # self.feas_ref_dict=self.feas_ref_gen.solve()
        # self.feas_ref_states=self.feas_ref_dict['z_opt']
        # self.feas_ref_inputs=self.feas_ref_dict['u_opt']

        self.control_prev = np.zeros((2,1))
        self.reference_regeneration()

        # Debugging: see the reference solution.
        # plt.subplot(411)
        # # import pdb; pdb.set_trace()
        # plt.plot(self.reference[:,0], self.reference[:,1], 'kx')
        # plt.plot(self.reference[:,0], self.feas_ref_states[:,0], 'r')

        # plt.ylabel('x')
        # plt.subplot(412)
        # plt.plot(self.reference[:,0], self.reference[:,2], 'kx')
        # plt.plot(self.reference[:,0], self.feas_ref_states[:,1], 'r')
        # plt.ylabel('y')
        # plt.subplot(413)
        # plt.plot(self.reference[:,0], self.reference[:,3], 'kx')
        # plt.plot(self.reference[:,0], self.feas_ref_states[:,2], 'r')
        # plt.ylabel('yaw')
        # plt.subplot(414)
        # plt.plot(self.reference[:,0], self.reference[:,4], 'kx')
        # plt.plot(self.reference[:,0], self.feas_ref_states[:,3], 'r')
        # plt.ylabel('v')

        # plt.figure()
        # plt.subplot(211)
        # plt.plot(self.reference[:-1,0], self.feas_ref_inputs[:,0])
        # plt.ylabel('acc')
        # plt.subplot(212)
        # plt.plot(self.reference[:-1,0], self.feas_ref_inputs[:,1])
        # plt.ylabel('df')
        # plt.show()

        # pdb.set_trace()
        # MPC initialization (might take a while....)
        if not self.ol_flag:
            self.SMPC=smpc.SMPC_MMPreds(N=self.N, DT=dt, N_modes_MAX=self.N_modes, NS_BL_FLAG=self.ns_bl_flag,
                                        L_F=self.lf, L_R=self.lr)
        else:
            self.SMPC=smpc.SMPC_MMPreds_OL(N=self.N, DT=dt, N_modes_MAX=self.N_modes,
                                          L_F=self.lf, L_R=self.lr)


        self.goal_reached = False # flags when the end of the path is reached and agent should stop




    def fit_velocity_profile(self):
        t_fits = [0.]
        traj = self.frenet_traj.trajectory

        for state, next_state in zip(traj[:-1, :], traj[1:, :]):
            s, x, y, yaw, curv = state
            sn, xn, yn, yawn, curvn = next_state

            v_curr = min( self.nominal_speed, np.sqrt(self.lat_accel_max / max(0.01, np.abs(curv))) )

            t_fits.append( (sn - s) / v_curr + t_fits[-1] )

        # Interpolate the points at time discretization dt.
        t_disc    = np.arange(t_fits[0], t_fits[-1] + self.dt/2, self.dt)
        s_disc    = np.interp(t_disc, t_fits, traj[:,0])
        x_disc    = np.interp(t_disc, t_fits, traj[:,1])
        y_disc    = np.interp(t_disc, t_fits, traj[:,2])
        yaw_disc  = np.interp(t_disc, t_fits, traj[:,3])
        curv_disc = np.interp(t_disc, t_fits, traj[:,4])
        v_disc    = np.diff(s_disc) / np.diff(t_disc)
        v_disc    = np.insert(v_disc, -1, v_disc[-1]) # repeat the last speed

        # v_int    = filtfilt(np.ones((3,))/3,1,v_disc)
        # curv_int = filtfilt(np.ones((3,))/3,1,curv_disc)

        # s_int = [ s_disc[0] ]
        # x_int = [ x_disc[0] ]
        # y_int = [ y_disc[0] ]
        # yaw_int = [ yaw_disc[0] ]

        # for vel in v_int:
        #     ds = vel * self.dt
        #     curr_s = s_int[-1] + ds
        #     curv = np.interp(curr_s, s_disc, curv_disc)

        #     dx = np.cos(yaw_int[-1]) * ds
        #     dy = np.sin(yaw_int[-1]) * ds
        #     dyaw = curv * ds

        #     s_int.append( curr_s )
        #     x_int.append( x_int[-1] + dx )
        #     y_int.append( y_int[-1] + dy )
        #     yaw_int.append( yaw_int[-1] + dyaw )

        # s_int = np.array(s_int[:-1])
        # x_int = np.array(x_int[:-1])
        # y_int = np.array(y_int[:-1])
        # yaw_int = np.array(yaw_int[:-1])

        # plt.subplot(211)
        # plt.plot(t_disc, v_disc, 'rx')
        # plt.plot(t_disc, v_int, 'b')
        # plt.subplot(212)
        # plt.plot(t_disc, curv_disc, 'rx')
        # plt.plot(t_disc, curv_int, 'b')

        # plt.figure()
        # plt.subplot(411)
        # plt.plot(t_disc, s_disc, 'rx')
        # plt.plot(t_disc, s_int, 'b')

        # plt.subplot(412)
        # plt.plot(t_disc, x_disc, 'rx')
        # plt.plot(t_disc, x_int, 'b')

        # plt.subplot(413)
        # plt.plot(t_disc, y_disc, 'rx')
        # plt.plot(t_disc, y_int, 'b')

        # plt.subplot(414)
        # plt.plot(t_disc, yaw_disc, 'rx')
        # plt.plot(t_disc, yaw_int, 'b')

        # plt.show()


        self.reference = np.column_stack((t_disc, x_disc, y_disc, yaw_disc, v_disc))
        # self.reference = np.column_stack((t_disc, x_int, y_int, yaw_int, v_int))


    def reference_regeneration(self, *state):
        if self.time==0:
            init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
            goal          = self.map.get_waypoint(self.goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
            route = self.planner.trace_route(init_waypoint.transform.location, goal.transform.location)

            way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
            self.frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)
            self.nominal_speed = self.nominal_speed_mps
            self.lat_accel_max = 2.0 # maximum lateral acceleration (m/s^2), for slowing down at turns

            self.fit_velocity_profile()

            self.ref_horizon= self.reference.shape[0]-1
            self.ref_dict={'x_ref':self.reference[1:,1], 'y_ref':self.reference[1:,2], 'psi_ref':self.reference[1:,3], 'v_ref':self.reference[1:,4],
                            'x0'  : self.reference[0,1],  'y0'  : self.reference[0,2],  'psi0'  : self.reference[0,3],  'v0'  : self.reference[0,4], 'acc_prev' : self.control_prev[0], 'df_prev' : self.control_prev[1]}
            self.ref_dict['psi_ref'] = fth.fix_angle( self.ref_dict['psi_ref'] - self.ref_dict['psi0']) + self.ref_dict['psi0']
            self.feas_ref_gen=smpc.RefTrajGenerator(N=self.ref_horizon, DT=self.dt, L_F=self.lf, L_R=self.lr)
            self.feas_ref_gen.update(self.ref_dict)
            self.feas_ref_dict=self.feas_ref_gen.solve()
            self.feas_ref_states=self.feas_ref_dict['z_opt']
            self.feas_ref_states=np.vstack((self.feas_ref_states, np.array([self.feas_ref_states[-1,:]]*(self.N+1))))
            self.feas_ref_inputs=self.feas_ref_dict['u_opt']
            self.feas_ref_inputs=np.vstack((self.feas_ref_inputs, np.array([self.feas_ref_inputs[-1,:]]*self.N)))
            self.feas_ref_states_new=self.feas_ref_states
            self.feas_ref_inputs_new=self.feas_ref_inputs
            # plt.plot(-self.feas_ref_states_new[:,1], self.feas_ref_states_new[:,0], 'kx', -self.feas_ref_states[:,1], self.feas_ref_states[:,0], 'ro')
            # plt.plot(-self.reference[:,2], self.reference[:,1], 'r-')
            # plt.axis("equal")
            # plt.show()
            # pdb.set_trace()
        else:

            x,y,psi,speed=state

            self.feas_ref_gen=smpc.RefTrajGenerator(N=self.ref_horizon-self.t_ref-1, DT=self.dt, L_F=self.lf, L_R=self.lr)

            self.ref_dict={'x_ref':self.feas_ref_states[self.t_ref+1:self.ref_horizon,0], 'y_ref':self.feas_ref_states[self.t_ref+1:self.ref_horizon,1], 'psi_ref':self.feas_ref_states[self.t_ref+1:self.ref_horizon,2], 'v_ref':self.feas_ref_states[self.t_ref+1:self.ref_horizon,3],
                            'x0'  : x,  'y0'  : y,  'psi0'  : psi,  'v0'  : speed, 'acc_prev' : self.control_prev[0], 'df_prev' : self.control_prev[1]}
            self.ref_dict['psi_ref'] = fth.fix_angle( self.ref_dict['psi_ref'] - self.ref_dict['psi0']) + self.ref_dict['psi0']
            self.feas_ref_gen.update(self.ref_dict)
            self.feas_ref_dict=self.feas_ref_gen.solve()
            self.feas_ref_states_new=self.feas_ref_dict['z_opt']

            self.feas_ref_states_new=np.vstack((self.feas_ref_states_new, np.array([self.feas_ref_states_new[-1,:]]*(self.N+1))))
            self.feas_ref_inputs_new=self.feas_ref_dict['u_opt']
            print(self.feas_ref_inputs_new.shape)
            if len(self.feas_ref_inputs_new.shape)!=1:
                self.feas_ref_inputs_new=np.vstack((self.feas_ref_inputs_new, np.array([self.feas_ref_inputs_new[-1,:]]*self.N)))
            else:
                self.feas_ref_inputs_new=np.array([self.feas_ref_inputs_new]*(self.N+1)).reshape(self.N+1,2)
            # plt.plot(-self.feas_ref_states_new[:,1], self.feas_ref_states_new[:,0], 'kx', -self.feas_ref_states[:,1], self.feas_ref_states[:,0], 'ro')
            # plt.show()


    def done(self):
        return self.goal_reached

    def run_step(self, pred_dict):
        vehicle_loc   = self.vehicle.get_location()
        vehicle_wp    = self.map.get_waypoint(vehicle_loc)
        vehicle_tf    = self.vehicle.get_transform()
        vehicle_vel   = self.vehicle.get_velocity()
        vehicle_accel = self.vehicle.get_acceleration()
        speed_limit   = self.vehicle.get_speed_limit()

        target_vehicle_positions=pred_dict["tvs_positions"]
        target_vehicle_gmm_preds=pred_dict["tvs_mode_dists"]

        # Get the vehicle's current pose in a RH coordinate system.
        x, y = vehicle_loc.x, -vehicle_loc.y
        psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))

        # Look up the projection of the current pose to Frenet frame.
        s, ey, epsi = \
            self.frenet_traj.convert_global_to_frenet_frame(x, y, psi)
        curv = self.frenet_traj.get_curvature_at_s(s)

        # Get the current speed and longitudinal acceleration.
        speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
        accel = np.cos(psi) * vehicle_accel.x - np.sin(psi)*vehicle_accel.y

        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False

        z0=np.array([x,y,psi,speed])
        u0=np.array([self.SMPC.A_MIN, 0.])
        v_des = np.clip(z0[-1] + self.SMPC.A_MIN * self.SMPC.DT, self.SMPC.V_MIN, self.SMPC.V_MAX)
        is_opt=False
        solve_time=np.nan


        self.t_ref=np.argmin(np.linalg.norm(self.feas_ref_states[:,:2]-np.hstack((x,y)), axis=1))


        self.time+=1
        """ TODO: add in SMPC code """
        if self.goal_reached or self.frenet_traj.reached_trajectory_end(s, resolution=5.):#or self.t_ref>self.ref_horizon-self.SMPC.N-1:
            # Stop if the end of the path is reached and signal completion.
            self.goal_reached = True

        else:
            # Run SMPC Preds.
            if self.time%60==0 and self.time>0 and self.ref_horizon>self.t_ref+1:
                self.reference_regeneration(x,y,psi,speed)

            t_ref_new=np.argmin(np.linalg.norm(self.feas_ref_states_new[:,:2]-np.hstack((x,y)), axis=1))
            # pdb.set_trace()
            update_dict={  'dx0':x-self.feas_ref_states_new[t_ref_new,0],     'dy0':y-self.feas_ref_states_new[t_ref_new,1],         'dpsi0':psi-self.feas_ref_states_new[t_ref_new,2],       'dv0':speed-self.feas_ref_states_new[t_ref_new,3],
                          'x_tv0': [target_vehicle_positions[k][0] for k in range(len(target_vehicle_positions))],        'y_tv0': [target_vehicle_positions[k][1] for k in range(len(target_vehicle_positions))],
                         'x_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,0].T,      'y_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,1].T,     'psi_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,2].T,   'v_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,3].T,
                         'a_ref': self.feas_ref_inputs_new[t_ref_new:t_ref_new+self.SMPC.N,0].T,       'df_ref': self.feas_ref_inputs_new[t_ref_new:t_ref_new+self.SMPC.N,1].T,
                         'mus'  : target_vehicle_gmm_preds[0],     'sigmas' : target_vehicle_gmm_preds[1], 'acc_prev' : self.control_prev[0], 'df_prev' : self.control_prev[1]}
            # update_dict={  'dx0':x-self.feas_ref_states[self.t_ref,0],     'dy0':y-self.feas_ref_states[self.t_ref,1],         'dpsi0':psi-self.feas_ref_states[self.t_ref,2],       'dv0':speed-self.feas_ref_states[self.t_ref,3],
            #             'x_tv0': [target_vehicle_positions[k][0] for k in range(len(target_vehicle_positions))],        'y_tv0': [target_vehicle_positions[k][1] for k in range(len(target_vehicle_positions))],
            #            'x_ref': self.feas_ref_states[self.t_ref:self.t_ref+self.SMPC.N+1,0].T,      'y_ref': self.feas_ref_states[self.t_ref:self.t_ref+self.SMPC.N+1,1].T,     'psi_ref': self.feas_ref_states[self.t_ref:self.t_ref+self.SMPC.N+1,2].T,   'v_ref': self.feas_ref_states[self.t_ref:self.t_ref+self.SMPC.N+1,3].T,
            #            'a_ref': self.feas_ref_inputs[self.t_ref:self.t_ref+self.SMPC.N,0].T,       'df_ref': self.feas_ref_inputs[self.t_ref:self.t_ref+self.SMPC.N,1].T,
            #            'mus'  : target_vehicle_gmm_preds[0],     'sigmas' : target_vehicle_gmm_preds[1]}

            if self.ol_flag:

                self.SMPC.update(update_dict)
                sol_dict=self.SMPC.solve()

                u_control = sol_dict['u_control'] # 2x1 vector, [a_optimal, df_optimal]
                v_next    = sol_dict['v_next']
                is_opt    = sol_dict['optimal']
                solve_time=sol_dict['solve_time']


            else:

                N_TV=len(target_vehicle_positions)
                t_bar=4
                i=(N_TV-1)*(self.SMPC.t_bar_max+1)+t_bar
                self.SMPC.update(i, update_dict)
                sol_dict=self.SMPC.solve(i)

                u_control = sol_dict['u_control'] # 2x1 vector, [a_optimal, df_optimal]
                v_next    = sol_dict['v_next']
                is_opt=sol_dict['optimal']
                solve_time=sol_dict['solve_time']

            self.control_prev=np.array([u_control[0]+update_dict['a_ref'][0],u_control[1]+update_dict['df_ref'][0]])
            u0=self.control_prev
            v_des=v_next

            print(f"\toptimal?: {is_opt}")
            print(f"\tv_next: {v_next}")
            print(f"\tsteering: {u_control[1]+update_dict['df_ref'][0]}")
            print(f"\tsolve time: {solve_time}")
            print(self.t_ref, self.time)

        control = self._low_level_control.update(speed,      # v_curr
                                                 u0[0], # a_des
                                                 v_des, # v_des
                                                 u0[1]) # df_des

        return control, z0, u0, is_opt, solve_time