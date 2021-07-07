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

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from utils import frenet_trajectory_handler as fth

class FrenetPIDAgent(object):
	""" A basic path following agent that doesn't do collision avoidance.
	    Can be extended for more sophisticated agents.
	"""

	def __init__(self, vehicle, carla_map, goal_location):
		self.vehicle = vehicle
		self.map     = carla_map
		self.planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(self.map, sampling_resolution=0.5) )
		self.planner.setup()

		# Get the high-level route using Carla's API (basically A* search over road segments).
		init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
		goal          = self.map.get_waypoint(goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
		route = self.planner.trace_route(init_waypoint.transform.location, goal.transform.location)

		# Convert the high-level route into a path parametrized by arclength distance s (i.e. Frenet frame).
		way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
		self._frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)

		# Control setup and parameters.
		self.control_prev = carla.VehicleControl()
		self.max_steer_angle = np.radians( self.vehicle.get_physics_control().wheels[0].max_steer_angle )
		self.alpha         = 0.8 # low-pass filter on actuation to simulate first order delay
		self.k_v           = 0.1 # P gain on velocity tracking error (throttle/brake)
		self.k_ey          = 0.5 # P gain on lateral error (steering)
		self.x_la          = 5.0 # lookahead distance for heading error (steering)
		self.lat_accel_max = 2.0 # maximum lateral acceleration (m/s^2), for slowing down at turns

		self.goal_reached = False # flags when the end of the path is reached and agent should stop

	def done(self):
		return self.goal_reached

	def run_step(self):
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
			self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
		curv = self._frenet_traj.get_curvature_at_s(s)

		# Get the current speed and longitudinal acceleration.
		speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
		accel = np.cos(psi) * vehicle_accel.x - np.sin(psi)*vehicle_accel.y

		control = carla.VehicleControl()
		control.hand_brake = False
		control.manual_gear_shift = False

		if self.goal_reached or self._frenet_traj.reached_trajectory_end(s):
			# Stop if the end of the path is reached and signal completion.
			self.goal_reached = True
			control.throttle = 0.
			control.brake    = -1.
		else:
			# Generate reference by identifying a max speed based on curvature + stoplights.
			if np.abs(curv) > 0.01:
				max_speed = 3.6 * np.sqrt(self.lat_accel_max / np.abs(curv))
				max_speed = min(max_speed, speed_limit)
			else:
				max_speed = speed_limit

			# Longitudinal control with hysteresis.
			if speed > max_speed + 2.0:
				control.throttle = 0.0
				control.brake = self.k_v * (speed - max_speed)
			elif speed < max_speed - 2.0:
				control.throttle = self.k_v * (max_speed - speed)
				control.brake    = 0.0
			else:
				control.throttle = 0.1
				control.brake    = 0.0

			# Simulated actuation delay, also used to avoid high frequency control inputs.
			if control.throttle > 0.0:
				control.throttle = self.alpha * control.throttle + (1. - self.alpha) * self.control_prev.throttle

			elif control.brake > 0.0:

				control.brake    = self.alpha * control.brake    + (1. - self.alpha) * self.control_prev.brake

		# Steering control: just using feedback for now, could add feedforward based on curvature.
		control.steer    = self.k_ey * (ey + self.x_la * epsi) / self.max_steer_angle
		control.steer    = self.alpha * control.steer    + (1. - self.alpha) * self.control_prev.steer

		# Clip Carla control to limits.
		control.throttle = np.clip(control.throttle, 0.0, 1.0)
		control.brake    = np.clip(control.brake, 0.0, 1.0)
		control.steer    = np.clip(control.steer, -1.0, 1.0)

		self.control_prev = control
		return control