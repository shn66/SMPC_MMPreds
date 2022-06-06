import carla
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.signal import filtfilt
from matplotlib.patches import Ellipse

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
                 smpc_config = "full",
                 OAIA=False,
                 obca=False,
                 obca_mode=2,
                 fps=20
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
        self.fps=fps
        self.d_min=2.0

        self.OA_inner_approx=OAIA

        self.obca_flag=obca
        self.obca_mode=obca_mode

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





        self.control_prev = np.zeros((2,1))
        self.prev_opt=False
        self.prev_nom_inputs=[]
        self.reference_regeneration()

        self.warm_start={}

        # Debugging: see the reference solution.

        plt.subplot(411)
        # import pdb; pdb.set_trace()
        plt.plot(self.reference[:,0], self.reference[:,1], 'kx')
        plt.plot(self.reference[:,0], self.feas_ref_states[:self.reference.shape[0],0], 'r')

        plt.ylabel('x')
        plt.subplot(412)
        plt.plot(self.reference[:,0], self.reference[:,2], 'kx')
        plt.plot(self.reference[:,0], self.feas_ref_states[:self.reference.shape[0],1], 'r')
        plt.ylabel('y')
        plt.subplot(413)
        plt.plot(self.reference[:,0], self.reference[:,3], 'kx')
        plt.plot(self.reference[:,0], self.feas_ref_states[:self.reference.shape[0],2], 'r')
        plt.ylabel('yaw')
        plt.subplot(414)
        plt.plot(self.reference[:,0], self.reference[:,4], 'kx')
        plt.plot(self.reference[:,0], self.feas_ref_states[:self.reference.shape[0],3], 'r')
        plt.ylabel('v')

        plt.figure()
        plt.subplot(211)
        plt.plot(self.reference[:-1,0], self.feas_ref_inputs[:self.reference.shape[0]-1,0])
        plt.ylabel('acc')
        plt.subplot(212)
        plt.plot(self.reference[:-1,0], self.feas_ref_inputs[:self.reference.shape[0]-1,1])
        plt.ylabel('df')
        plt.show()

        # MPC initialization (might take a while....)
        if not self.ol_flag:
            if not self.obca_flag:
                self.SMPC=smpc.SMPC_MMPreds(N=self.N, DT=self.dt, N_modes_MAX=self.N_modes, NS_BL_FLAG=self.ns_bl_flag,
                                        L_F=self.lf, L_R=self.lr, fps=self.fps)
            else:
                self.SMPC=smpc.SMPC_MMPreds_OBCA(N=self.N, DT=self.dt, N_modes_MAX=self.N_modes, NS_BL_FLAG=self.ns_bl_flag,
                                        L_F=self.lf, L_R=self.lr, fps=self.fps, pol_mode=self.obca_mode)
        else:
            self.SMPC=smpc.SMPC_MMPreds_OL(N=self.N, DT=self.dt, N_modes_MAX=self.N_modes,
                                          L_F=self.lf, L_R=self.lr, fps=self.fps)


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




        self.reference = np.column_stack((t_disc, x_disc, y_disc, yaw_disc, v_disc))


    def reference_regeneration(self, *state):
        if self.time==0:
            # Get the high-level route using Carla's API (basically A* search over road segments).

            init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
            goal          = self.map.get_waypoint(self.goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
            route = self.planner.trace_route(init_waypoint.transform.location, goal.transform.location)

            # # Convert the high-level route into a path parametrized by arclength distance s (i.e. Frenet frame).
            # # Generate a refernece by fitting a velocity profile with specified nominal speed and time discretization.

            way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
            self.frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)
            self.nominal_speed = self.nominal_speed_mps
            self.lat_accel_max = 2. # maximum lateral acceleration (m/s^2), for slowing down at turns

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
            self.feas_ref_inputs=np.vstack((self.feas_ref_inputs, np.array([self.feas_ref_inputs[-1,:]]*(self.N+1))))
            self.feas_ref_states_new=self.feas_ref_states
            self.feas_ref_inputs_new=self.feas_ref_inputs

        else:

            x,y,psi,speed=state
            self.feas_ref_states_new=[]
            self.feas_ref_inputs_new=[]



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
                self.feas_ref_inputs_new=np.vstack((self.feas_ref_inputs_new, np.array([self.feas_ref_inputs_new[-1,:]]*(self.N+1)))).reshape((-1,2))
            else:
                self.feas_ref_inputs_new=np.array([self.feas_ref_inputs_new]*(self.N+1)).reshape((self.N+1,2))

            self.feas_ref_states_new=self.feas_ref_states_new
            self.feas_ref_inputs_new=self.feas_ref_inputs_new


    def linearization_traj(self, *state):
                x,y,psi,speed=state
                states=[np.array([x,y,psi,speed])]
                for t in range(self.N):
                    if t==self.N-1:
                        control=self.prev_nom_inputs[0][:,-1]
                    else:
                        control=self.prev_nom_inputs[0][:,t+1]
                    beta=np.arctan((self.lr/(self.lr+self.lf)*np.tan(control[1])))
                    x_next=states[t][0]+self.dt*(states[t][3]*np.cos(states[t][2]+beta))
                    y_next=states[t][1]+self.dt*(states[t][3]*np.sin(states[t][2]+beta))
                    psi_next=states[t][2]+self.dt*(states[t][3]/self.lr*np.sin(beta))
                    v_next  =states[t][3]+self.dt*control[0]
                    states.append(np.array([x_next,y_next,psi_next,v_next]))

                l_states=np.array(states).reshape((self.N+1,-1))
                l_inputs=self.prev_nom_inputs[0][:,1:].T
                l_inputs=np.vstack((l_inputs,np.array([l_inputs[-1,:]]*2)))

                return l_states, l_inputs



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

        N_TV=len(target_vehicle_positions)


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



        if self.goal_reached or self.frenet_traj.reached_trajectory_end(s, resolution=5.):
            # Stop if the end of the path is reached and signal completion.
            self.goal_reached = True

        else:
            # Run SMPC Preds.
            if self.time%20==0 and self.ref_horizon>self.t_ref+1:
                self.reference_regeneration(x,y,psi,speed)
                # if self.feas_ref_inputs_new.shape[0]==13:
                #     pdb.set_trace()



            t_ref_new=np.argmin(np.linalg.norm(self.feas_ref_states_new[:,:2]-np.hstack((x,y)), axis=1))
            if self.prev_opt and self.time%10==0:
                l_states, l_inputs = self.linearization_traj(x,y,psi,speed)

            else:
                l_states=self.feas_ref_states_new[t_ref_new:t_ref_new+self.N+1,:]
                l_inputs=self.feas_ref_inputs_new[t_ref_new:t_ref_new+self.N+1,:]


            ## TV shapes estimate along prediction horizon

            Rs_ev=[np.array([[np.cos(l_states[t,2]),np.sin(l_states[t,2])],[-np.sin(l_states[t,2]), np.cos(l_states[t,2])]]) for t in range(1,self.N+1)]


            tv_theta=[[np.arctan2(np.diff(target_vehicle_gmm_preds[k][0][j,:,1]), np.diff(target_vehicle_gmm_preds[k][0][j,:,0])) for j in range(self.N_modes)] for k in range(N_TV)]
            tv_R=[[[np.array([[np.cos(tv_theta[k][j][i]), np.sin(tv_theta[k][j][i])],[-np.sin(tv_theta[k][j][i]), np.cos(tv_theta[k][j][i])]]) for i in range(self.N-1)] for j in range(self.N_modes)] for k in range(N_TV)]
            if self.OA_inner_approx:
                tv_Q=np.array([[1./(3.6+self.d_min)**2, 0.],[0., 1./(1.2+self.d_min)**2]])
                tv_shape_matrices=[[[ tv_R[k][j][i].T@tv_Q@tv_R[k][j][i] for i in range(self.N-1)] for j in range(self.N_modes)] for k in range(N_TV)]
            elif not self.obca_flag:
                v_Q=np.array([[1./(2.6)**2, 0.],[0., 1./(1.45)**2]])
                tv_shape_matrices=[[[ np.identity(2) for i in range(self.N-1)] for j in range(self.N_modes)] for k in range(N_TV)]
                for k in range(N_TV):
                    for j in range(self.N_modes):
                        for i in range(self.N-1):
                            m_eval, m_evec= np.linalg.eigh(Rs_ev[i].T@v_Q@Rs_ev[i])
                            m_sqrt=m_evec@np.diag(np.sqrt(m_eval))@m_evec.T
                            m_sqrt_inv=m_evec@np.diag(np.sqrt(m_eval)**(-1))@m_evec.T
                            s_eval, s_evec= np.linalg.eigh(m_sqrt_inv@tv_R[k][j][i].T@v_Q@tv_R[k][j][i]@m_sqrt_inv)
                            temp=s_evec@np.diag(np.power(np.sqrt(s_eval)**(-1)+1., 2)**(-1))@s_evec.T
                            tv_shape_matrices[k][j][i]=m_sqrt@temp@m_sqrt
            else:
                tv_shape_matrices=tv_R

            update_dict={  'dx0':x-l_states[0,0],     'dy0':y-l_states[0,1],         'dpsi0':psi-l_states[0,2],       'dv0':speed-l_states[0,3],
                         'x_tv0': [target_vehicle_positions[k][0] for k in range(N_TV)],        'y_tv0': [target_vehicle_positions[k][1] for k in range(N_TV)],
                         'x_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,0].T,
                         'y_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,1].T ,
                         'psi_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,2].T ,
                         'v_ref': self.feas_ref_states_new[t_ref_new:t_ref_new+self.SMPC.N+1,3].T ,
                         'a_ref': self.feas_ref_inputs_new[t_ref_new:t_ref_new+self.SMPC.N+1,0].T ,
                         'df_ref': self.feas_ref_inputs_new[t_ref_new:t_ref_new+self.SMPC.N+1,1].T ,
                         'x_lin': l_states[:,0].T,
                         'y_lin': l_states[:,1].T ,
                         'psi_lin': l_states[:,2].T,
                         'v_lin': l_states[:,3].T ,
                         'a_lin': l_inputs[:,0].T ,
                         'df_lin': l_inputs[:,1].T,
                         'mus'  : [target_vehicle_gmm_preds[k][0] for k in range(N_TV)],     'sigmas' : [target_vehicle_gmm_preds[k][1] for k in range(N_TV)], 'acc_prev' : self.control_prev[0], 'df_prev' : self.control_prev[1],       'tv_shapes': tv_shape_matrices, 'Rs_ev': Rs_ev }



            if 'ws' in self.warm_start.keys() and self.obca_flag:
                update_dict.update({'ws': self.warm_start['ws']})



            if self.ol_flag:

                self.SMPC.update(update_dict)
                sol_dict=self.SMPC.solve()

                u_control = sol_dict['u_control'] # 2x1 vector, [a_optimal, df_optimal]
                v_next    = sol_dict['v_next']
                is_opt    = sol_dict['optimal']
                solve_time=sol_dict['solve_time']


            else:


                t_bar=4
                i=(N_TV-1)*(self.SMPC.t_bar_max)+t_bar
                self.SMPC.update(i, update_dict)
                sol_dict=self.SMPC.solve(i)

                u_control = sol_dict['u_control'] # 2x1 vector, [a_optimal, df_optimal]
                v_next    = sol_dict['v_next']
                is_opt=sol_dict['optimal']
                solve_time=sol_dict['solve_time']
                self.warm_start={}
                if is_opt and self.obca_flag:
                    self.warm_start={'ws': [sol_dict['h_opt'],sol_dict['K_opt'],sol_dict['M_opt'],sol_dict['lmbd_opt'],sol_dict['nu_opt']]}
                self.prev_opt=False
                # self.prev_opt=is_opt
                if self.prev_opt:
                    self.prev_nom_inputs=sol_dict['nom_u_ev']


            self.control_prev=np.array([u_control[0]+update_dict['a_lin'][0],u_control[1]+update_dict['df_lin'][0]])
            u0=self.control_prev
            v_des=v_next

            # scaled_distance=np.array([x+speed*np.cos(psi)*self.SMPC.DT-target_vehicle_gmm_preds[0][0][0,0,0],y+speed*np.sin(psi)*self.SMPC.DT-target_vehicle_gmm_preds[0][0][0,0,1]]).T@tv_shape_matrices[0][0][0]@np.array([x+speed*np.cos(psi)*self.SMPC.DT-target_vehicle_gmm_preds[0][0][0,0,0],y+speed*np.sin(psi)*self.SMPC.DT-target_vehicle_gmm_preds[0][0][0,0,1]])


            print(f"\toptimal?: {is_opt}")
            print(f"\tv_next: {v_next}")
            print(f"\tsteering: {u0[1]}")

            # print(f"\tsolve time: {solve_time}")
            # print(f"\t scaled distance_x: {np.sqrt(scaled_distance)}")
            # print(self.t_ref, self.time)


            ## Debugging: Plot expected hyperplanes for obstacle avoidance along the prediction horizon

            # if self.time%10 ==0 and is_opt:
            #     for i, c in zip( range(len(sol_dict["nom_z_ev"])), ['r', 'g', 'b']):
            #         arr = sol_dict["nom_z_ev"][i]
            #         arr_lin=sol_dict["z_lin"]
            #         arr_ref=sol_dict['z_ref']
            #         arr_tv= sol_dict['z_tv_ref']
            #         shape=tv_shape_matrices[0][0]
            #         # pdb.set_trace()
            #         # x_ref=[arr_tv[0,t+1]+(x-arr_tv[0,t+1])/np.sqrt((np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]])).T@shape[t]@(np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]]))) for t in range(self.N)]
            #         # y_ref=[arr_tv[0,t+1]+(x-arr_tv[0,t+1])/np.sqrt((np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]])).T@shape[t]@(np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]]))) for t in range(self.N)]

            #         plt.subplot(3, 1, 1+i)
            #         plt.legend()
            #         plt.plot(arr[0,:], arr[1,:], color=c, marker='*')
            #         plt.plot(arr_lin[0,:], arr_lin[1,:], 'k-')
            #         plt.plot(arr_ref[0,:], arr_ref[1,:], 'k.')
            #         plt.plot(arr_tv[0,:], arr_tv[1,:], 'y.')
            #         delx=0.06*(np.arange(10)-4.5)
            #         for t in range(self.N):

            #             if t==self.N-1:
            #                 shapet=shape[t-1]
            #                 theta_el=tv_theta[0][0][t-1]
            #             else:
            #                 print(sol_dict['eval_oa'][t,:]@(arr[:2,t]-arr_tv[:,t+1])-1.0)
            #                 shapet=shape[t]
            #                 theta_el=tv_theta[0][0][t]

            #             x_ref=arr_tv[0,t+1]+(x-arr_tv[0,t+1])/np.sqrt((np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]])).T@shapet@(np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]])))
            #             y_ref=arr_tv[1,t+1]+(y-arr_tv[1,t+1])/np.sqrt((np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]])).T@shapet@(np.array([x-arr_tv[0,t+1], y-arr_tv[1,t+1]])))
            #             zQ=np.array([x_ref-arr_tv[0,t+1], y_ref-arr_tv[1,t+1]]).T@shapet
            #             x_plt=delx+x_ref
            #             y_plt=(-zQ[0]*delx)/zQ[1]+y_ref
            #             # pdb.set_trace()
            #             plt.plot(x_plt, y_plt, label ='%s line' % t)
            #             plt.plot([x, arr_tv[0,t+1]], [y, arr_tv[1,t+1]], 'r--')
            #             plt.arrow(x_ref, y_ref, zQ[0]*1, zQ[1]*1)

            #             ax = plt.gca()
            #             ax.add_patch(Ellipse((arr_tv[0,t+1],
            #                                   arr_tv[1,t+1]),
            #                                   2*(3+self.d_min),
            #                                   2*(1+self.d_min),
            #                                   theta_el,
            #                                   fill=False,
            #                                   color='c')
            #                         )

            #         plt.axis('equal')
            #     plt.legend()
            #     plt.show()
                # pdb.set_trace()

        self.time+=1
        control = self._low_level_control.update(speed,      # v_curr
                                                 u0[0], # a_des
                                                 v_des, # v_des
                                                 u0[1]) # df_des

        return control, z0, u0, is_opt, solve_time