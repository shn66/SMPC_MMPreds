import carla
import os
import sys
import numpy as np

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

class SMPCAgent(object):
	""" Implementation of an agent using multimodal predictions and stochastic MPC for control. """

	def __init__(self,
		         vehicle,                  # Vehicle object that this agent controls
		         carla_map,                # Map object used to query waypoints from locations
		         goal_location,            # desired goal location used to generate a path
		         nominal_speed_mps = 12.0, # sets desired speed (m/s) for tracking path
		         dt =0.2                   # time discretization (s) used to generate a reference
		         ):
		self.vehicle = vehicle
		self.map     = carla_map
		self.planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(self.map, sampling_resolution=0.5) )
		self.planner.setup()

		# Get the high-level route using Carla's API (basically A* search over road segments).
		init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
		goal          = self.map.get_waypoint(goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
		route = self.planner.trace_route(init_waypoint.transform.location, goal.transform.location)

		# Convert the high-level route into a path parametrized by arclength distance s (i.e. Frenet frame).
		# Generate a refernece by fitting a velocity profile with specified nominal speed and time discretization.
		way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
		self.frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)
		self.nominal_speed = nominal_speed_mps
		self.lat_accel_max = 2.0 # maximum lateral acceleration (m/s^2), for slowing down at turns
		self.dt = dt
		self.fit_velocity_profile()

		# Feasible reference generation by nonlinear trajectory optimization
		self.feas_ref_gen=smpc.RefTrajGenerator(N=self.ref_horizon, DT=dt)
		self.feas_ref_gen.update(self.ref_dict)
		self.feas_ref_dict=feas_ref_gen.solve()
		self.feas_ref_states=self.feas_ref_dict['z_opt']
		self.feas_ref_inputs=self.feas_ref_dict['u_opt']

		# MPC initialization (might take a while....)
		self.SMPC=smpc.SMPC_MMPreds(DT=dt)


		# Control setup and parameters.
		self.control_prev = carla.VehicleControl()
		self.max_steer_angle = np.radians( self.vehicle.get_physics_control().wheels[0].max_steer_angle )
		self.alpha         = 0.8 # low-pass filter on actuation to simulate first order delay

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
		self.ref_horizon= self.reference.shape[0]
		self.ref_dict={'x_ref':self.reference[:,1], 'y_ref':self.reference[:,2], 'psi_ref':self.reference[:,3], 'v_ref':self.reference[:,4],
						'x0'  :self.reference[0,1],  'y0'  :self.reference[0,2],  'psi0'  :self.reference[0,3],  'v0'  :self.reference[0,4]}



	def done(self):
		return self.goal_reached

	def run_step(self, target_vehicle_positions, target_vehicle_gmm_preds):
		vehicle_loc   = self.vehicle.get_location()
		vehicle_wp    = self.map.get_waypoint(vehicle_loc)
		vehicle_tf    = self.vehicle.get_transform()
		vehicle_vel   = self.vehicle.get_velocity()
		vehicle_accel = self.vehicle.get_acceleration()
		speed_limit   = self.vehicle.get_speed_limit()

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

		""" TODO: add in SMPC code """
		if self.goal_reached or self.frenet_traj.reached_trajectory_end(s):
			# Stop if the end of the path is reached and signal completion.
			self.goal_reached = True
			control.throttle = 0.
			control.brake    = -1.
		else:
			# Run SMPC Preds.
			t_ref=np.argmin(np.linalg.norm(self.feas_ref_states[:,:2]-np.hstack((x,y)), axis=1))
			update_dict={  'dx0':x-self.feas_ref_states[t_ref,0],     'dy0':y-self.feas_ref_states[t_ref,1],         'dpsi0':psi-self.feas_ref_states[t_ref,2],       'dv0':speed-self.feas_ref_states[t_ref,3],
						  'xtv0': [target_vehicle_positions[k][:,0] for k in range(len(target_vehicle_positions))],        'ytv0': [target_vehicle_positions[k][:,1] for k in range(len(target_vehicle_positions))],
					 	 'x_ref': self.feas_ref_states[t_ref:t_ref+self.SMPC.N+1,0].T,      'y_ref': self.feas_ref_states[t_ref:t_ref+self.SMPC.N+1,1].T,     'psi_ref': self.feas_ref_states[t_ref:t_ref+self.SMPC.N+1,2].T,   'v_ref': self.feas_ref_states[t_ref:t_ref+self.SMPC.N+1,3].T,
					 	 'a_ref': self.feas_ref_inputs[t_ref:t_ref+self.SMPC.N,0].T,       'df_ref': self.feas_ref_inputs[t_ref:t_ref+self.SMPC.N,1].T,
					 	 'mus'  : target_vehicle_gmm_preds[0],     'sigmas' target_vehicle_gmm_preds[1]}
			N_TV=len(target_vehicle_positions)
			t_bar=3
			i=(N_TV-1)*(self.SMPC.t_bar_max+1)+t_bar
			self.SMPC.update(i, update_dict)
			sol_dict=self.SMPC.solve(i)

			u_control = sol_dict['u_control'] # 2x1 vector, [a_optimal, df_optimal]
			v_next    = sol_dict['v_next']
			
			control.throttle = 0.
			control.brake = 0.
			control.steer = 0.

		# Clip Carla control to limits.
		control.throttle = np.clip(control.throttle, 0.0, 1.0)
		control.brake    = np.clip(control.brake, 0.0, 1.0)
		control.steer    = np.clip(control.steer, -1.0, 1.0)

		self.control_prev = control
		return control