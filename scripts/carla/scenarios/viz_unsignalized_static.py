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
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)

from frenet_pid_agent import FrenetPIDAgent

#########################################################
### Scenario Setup (TODO: json).
# outer edges, i.e. may be in the wrong lane but should
# be approximately correct minus a required lane change.
# INTERSECTION = [\
# [[14.5, 6.0, 0], [43.1, 5.1, 0]],
# [[24.4, -16.3, 90], [24.6, 14.7, 90]],
# [[44.4, -5.2, 180], [15.2, -4.7, 180]],
# [[35.0, 15.0, 270], [35.0, -14.6, 270]]
# ]

INTERSECTION = [\
[[87.3, 5.5, 0], [115.0, 1.6, 0]],
[[96.3, -15.2, 90], [96.1, 14.1, 90]],
[[115.5, -2.2, 180], [88.1, -5.1, 180]],
[[106.7, 14.4, 270], [107.0, -13.7, 270]]
]

STATIC_CARS = [[0, 0, 'L'],  # facing east
               [3, 0, 'R'],  # facing north
               [2, 0, 'L']]  # facing west


# SCENARIO_CASE = 2
# DYNAMIC_CARS = []
# if SCENARIO_CASE == 0:
# 	DYNAMIC_CARS  = [[[0,0,'L'], [3,1,'L']],  # facing east, turn left towards north
# 	                 [[2,0,'R'], [2,1,'R']]]  # oncoming driving west
# elif SCENARIO_CASE == 1:
# 	DYNAMIC_CARS  = [[[0,0,'R'], [1,1,'R']],  # facing east, turn right towards south
# 	                 [[2,0,'L'], [1,1,'L']]]  # facing west, turning left towards south
# elif SCENARIO_CASE == 2:
# 	DYNAMIC_CARS  = [[[0,0,'L'], [3,1,'L']],  # facing east, turn left towards north
# 	                 [[2,0,'L'], [1,1,'L']]]  # oncoming driving west
# else:
# 	raise NotImplemented("That scenario has not been made yet.")

# COLORS        = ['186,0,0', '65,63,197'] # using colors from Audi.TT set.
# assert(len(DYNAMIC_CARS) == len(COLORS))
#########################################################

def make_transform_from_pose(pose, spawn_height=1.5):
	location = carla.Location( x=pose[0], y = pose[1], z=spawn_height)
	rotation = carla.Rotation(yaw=pose[2])
	return carla.Transform(location, rotation)

def shift_pose_along_lane(pose, shift_m=20):
	# This function shifts back the pose of a vehicle
	# by simply moving it forward relative to the orientation
	# of the vehicle by shift_m meters.

	forward_yaw_angle = np.radians(float(pose[2]))

	delta_x = shift_m * np.cos(forward_yaw_angle)
	delta_y = shift_m * np.sin(forward_yaw_angle)

	return [pose[0] + delta_x,
	        pose[1] + delta_y,
	        pose[2]]

def shift_pose_across_lane(pose, left_shift_m=3.7):
	# This function shifts the pose "laterally" to another lane.
	# If left_shift_m is positive, it will move the car to its left.

	left_yaw_angle = np.radians(float(pose[2])) - np.pi/2.

	delta_x = left_shift_m * np.cos(left_yaw_angle)
	delta_y = left_shift_m * np.sin(left_yaw_angle)

	return [pose[0] + delta_x,
	        pose[1] + delta_y,
	        pose[2]]

def setup_static_cars(world):
	static_vehicle_list = []

	bp_library = world.get_blueprint_library()
	npc_bp = bp_library.filter("vehicle.audi.tt")[0]
	# Recommended values for Audi.TT
	# ['186,0,0', '65,63,197', '67,67,67', '246,246,246', '230,221,0', '178,114,0']
	npc_bp.set_attribute('color', '186,0,0')

	for car_location_inds in STATIC_CARS:
		pose = INTERSECTION[car_location_inds[0]][car_location_inds[1]]
		if car_location_inds[2] == 'L':
			pose = shift_pose_across_lane(pose)

		npc_transform = make_transform_from_pose(pose)
		static_vehicle_list.append( world.spawn_actor(npc_bp, npc_transform) )

	return static_vehicle_list

def setup_dynamic_cars(world):
	dynamic_vehicle_list = []
	dynamic_policy_list  = []

	bp_library = world.get_blueprint_library()
	dyn_bp = bp_library.filter("vehicle.mercedes-benz.coupe")[0]

	town_map = world.get_map()

	random.seed(0) # setting deterministic sampling of vehicle colors.

	for start_goal, color in zip(DYNAMIC_CARS, COLORS):
		dyn_bp.set_attribute('color', color)
		start, goal = start_goal

		start_pose = shift_pose_along_lane(INTERSECTION[start[0]][start[1]], -10.)
		goal_pose  = shift_pose_along_lane(INTERSECTION[goal[0]][goal[1]], 10.)

		if start[2] == 'L':
			start_pose = shift_pose_across_lane(start_pose)

		if goal[2]  == 'L':
			goal_pose = shift_pose_across_lane(goal_pose)

		start_transform = make_transform_from_pose(start_pose)
		goal_transform  = make_transform_from_pose(goal_pose)

		veh_actor   = world.spawn_actor(dyn_bp, start_transform)

		veh_policy  = FrenetPIDAgent(veh_actor, town_map,  goal_transform.location)

		dynamic_vehicle_list.append(veh_actor)
		dynamic_policy_list.append(veh_policy)

	return dynamic_vehicle_list, dynamic_policy_list

def setup_camera(world):
	bp_library = world.get_blueprint_library()
	bp_drone  = bp_library.find('sensor.camera.rgb')

	# This is like a top down view of the intersection.  Can tune later.
	cam_loc = carla.Location(x=30., y=0., z=50.)
	cam_ori = carla.Rotation(pitch=-90, yaw=0., roll=0.)
	cam_transform = carla.Transform(cam_loc, cam_ori)

	bp_drone.set_attribute('image_size_x', str(1920))
	bp_drone.set_attribute('image_size_x', str(1080))
	bp_drone.set_attribute('fov', str(60))
	bp_drone.set_attribute('role_name', 'drone')
	drone = world.spawn_actor(bp_drone, cam_transform)

	return drone

def main():
	static_vehicle_list = []
	# dynamic_vehicle_list = []
	# dynamic_policy_list = []
	try:
		client = carla.Client("localhost", 2000)
		client.set_timeout(2.0)

		world = client.get_world()
		if world.get_map().name != "Town05":
			world = client.load_world("Town05")
		world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

		static_vehicle_list = setup_static_cars(world)
		# dynamic_vehicle_list, dynamic_policy_list = setup_dynamic_cars(world)
		drone = setup_camera(world)

		completed = False         # Flag to indicate when all cars have reached their destination
		fps = 20                  # FPS for the simulation under synchronous mode
		use_spectator_view = True # Flag to indicate whether to overwrite default drone view with spectator view
		opencv_viz = False        # Flag to indicate whether to create an external window to view the drone view
		save_avi   = True         # Flag to indicate whether to save an avi of the drone view.

		with CarlaSyncMode(world, drone, fps=fps) as sync_mode:
			if use_spectator_view:
				spectator_transform = world.get_spectator().get_transform()
				drone.set_transform(spectator_transform)

			# writer = None
			# if save_avi:
			# 	writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (960, 500))

			for _ in range(50):
				snap, img = sync_mode.tick(timeout=2.0)

			while not completed:
				snap, img = sync_mode.tick(timeout=2.0)

				# Handle drone view.
				img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
				img_array = np.reshape(img_array, (img.height, img.width, 4))
				img_array = img_array[:, :, :3]
				cv2.imwrite("intersection_ex.png", img_array)
				# img_array = cv2.resize(img_array, (960, 500), interpolation = cv2.INTER_AREA)

				# # Handle OpenCV stuff.
				# if opencv_viz:
				# 	cv2.imshow('Drone', img_array); cv2.waitKey(1)
				# if save_avi:
				# 	writer.write(img_array)

				# Handle updating the dynamic cars.  Terminate once all cars reach the goal.
				completed = True
				# for act, policy in zip(dynamic_vehicle_list, dynamic_policy_list):
				# 	control = policy.run_step(1./float(fps))
				# 	completed = completed and policy.done()
				# 	act.apply_control(control)

			# if save_avi:
			# 	writer.release()

	finally:
		for actor in static_vehicle_list:
			actor.destroy()

		# for actor in dynamic_vehicle_list:
		# 	actor.destroy()

		drone.destroy()

		cv2.destroyAllWindows()
		print('Done.')

if __name__ == '__main__':
	main()