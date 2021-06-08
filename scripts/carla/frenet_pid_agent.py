import carla
import os
import sys
import argparse
import cv2
import numpy as np
import random
import time

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
	raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)

sys.path.append(CARLA_ROOT + "PythonAPI/carla/agents/")
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import frenet_trajectory_handler as fth

"""
Adapted from the VehicleAgent in simulation.py.  Need to refactor that code.
"""

class FrenetPIDAgent(object):
	def __init__(self, vehicle, carla_map, goal_location):
		self.vehicle = vehicle
		self.map     = carla_map
		self.planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(self.map, sampling_resolution=0.5) )
		self.planner.setup()

		init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))

		goal          = self.map.get_waypoint(goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))

		route = self.planner.trace_route( init_waypoint.transform.location, goal.transform.location)

		way_s, way_xy, way_yaw = fth.extract_path_from_waypoints( route )
		self._frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)

		self.control_prev = carla.VehicleControl()
		self.max_steer_angle = np.radians( self.vehicle.get_physics_control().wheels[0].max_steer_angle )

		# Controller params:
		self.alpha = 0.8
		self.k_v = 0.1
		self.k_ey = 0.5
		self.x_la = 5.0

		self.goal_reached = False

	def done(self):
		return self.goal_reached

	def run_step(self, dt):
		vehicle_loc   = self.vehicle.get_location()
		vehicle_wp    = self.map.get_waypoint(vehicle_loc)
		vehicle_tf    = self.vehicle.get_transform()
		vehicle_vel   = self.vehicle.get_velocity()
		vehicle_accel = self.vehicle.get_acceleration()
		speed_limit   = self.vehicle.get_speed_limit()

		x, y = vehicle_loc.x, -vehicle_loc.y
		psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))

		s, ey, epsi = \
			self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
		curv = self._frenet_traj.get_curvature_at_s(s)

		speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
		accel = np.cos(psi) * vehicle_accel.x - np.sin(psi)*vehicle_accel.y

		control = carla.VehicleControl()
		control.hand_brake = False
		control.manual_gear_shift = False

		if self.goal_reached or self._frenet_traj.reached_trajectory_end(s):
			self.goal_reached = True
			control.throttle = 0.
			control.brake    = -1.
		else:
			# Step 1: Generate reference by identifying a max speed based on curvature + stoplights.
			# TODO: update logic, maybe use speed limits from Carla.
			lat_accel_max = 2.0 # m/s^2
			speed_limit   = 13.  # m/s -> 29 mph ~ 30 mph

			if np.abs(curv) > 0.01:
				max_speed = 3.6 * np.sqrt(lat_accel_max / np.abs(curv))
				max_speed = min(max_speed, speed_limit)
			else:
				max_speed = speed_limit

			if speed > max_speed + 2.0:
				control.throttle = 0.0
				control.brake = self.k_v * (speed - max_speed)
			elif speed < max_speed - 2.0:
				control.throttle = self.k_v * (max_speed - speed)
				control.brake    = 0.0
			else:
				control.throttle = 0.1
				control.brake    = 0.0

			if control.throttle > 0.0:
				control.throttle = self.alpha * control.throttle + (1. - self.alpha) * self.control_prev.throttle

			elif control.brake > 0.0:
				control.brake    = self.alpha * control.brake    + (1. - self.alpha) * self.control_prev.brake

		control.steer    = self.k_ey * (ey + self.x_la * epsi) / self.max_steer_angle
		control.steer    = self.alpha * control.steer    + (1. - self.alpha) * self.control_prev.steer

		control.throttle = np.clip(control.throttle, 0.0, 1.0)
		control.brake    = np.clip(control.brake, 0.0, 1.0)
		control.steer    = np.clip(control.steer, -1.0, 1.0)

		self.control_prev = control
		return control