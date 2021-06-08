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
INTERSECTION = [\
[[14.5, 6.0, 0], [43.1, 5.1, 0]],
[[24.4, -16.3, 90], [24.6, 14.7, 90]],
[[44.4, -5.2, 180], [15.2, -4.7, 180]],
[[35.0, 15.0, 270], [35.0, -14.6, 270]]
]

STATIC_CARS = [[1, 0], # facing south
               [3, 0]] # facing north

DYNAMIC_CARS  = [[[0,0], [3,1]],  # facing east, turn left towards north
                 [[2,0], [2,1]]]  # oncoming driving west
DYN_BEHAVIORS = ["normal", "normal"] # one of normal, cautious, aggressive
COLORS        = ['186,0,0', '65,63,197'] # using colors from Audi.TT set.
assert(len(DYN_BEHAVIORS) == len(DYNAMIC_CARS) == len(COLORS))
#########################################################

def make_transform_from_pose(pose, spawn_height=1.5):
	location = carla.Location( x=pose[0], y = pose[1], z=spawn_height)
	rotation = carla.Rotation(yaw=pose[2])
	return carla.Transform(location, rotation)

def shift_pose(pose, shift_m=20):
	# This function shifts back the pose of a vehicle
	# by simply moving it forward relative to the orientation
	# of the vehicle by shift_m meters.

	delta_x = shift_m * np.cos(np.radians(float(pose[2])))
	delta_y = shift_m * np.sin(np.radians(float(pose[2])))

	return [pose[0] + delta_x,
	        pose[1] + delta_y,
	        pose[2]]

def setup_static_cars(world):
	static_vehicle_list = []

	bp_library = world.get_blueprint_library()
	npc_bp = bp_library.filter("vehicle.audi.tt")[0]
	# Recommended values for Audi.TT
	# ['186,0,0', '65,63,197', '67,67,67', '246,246,246', '230,221,0', '178,114,0']
	npc_bp.set_attribute('color', '246,246,246')

	for car_location_inds in STATIC_CARS:
		pose = INTERSECTION[car_location_inds[0]][car_location_inds[1]]
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

	for start_goal, behavior_type, color in zip(DYNAMIC_CARS, DYN_BEHAVIORS, COLORS):
		dyn_bp.set_attribute('color', color)
		start, goal = start_goal

		start_pose = shift_pose(INTERSECTION[start[0]][start[1]], -10.)
		goal_pose  = shift_pose(INTERSECTION[goal[0]][goal[1]], 10.)

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
	cam_loc = carla.Location(x=30., y=0., z=100.)
	cam_ori = carla.Rotation(pitch=-90, yaw=0., roll=0.)
	cam_transform = carla.Transform(cam_loc, cam_ori)

	bp_drone.set_attribute('image_size_x', str(1920))
	bp_drone.set_attribute('image_size_x', str(1080))
	bp_drone.set_attribute('fov', str(90))
	bp_drone.set_attribute('role_name', 'drone')
	drone = world.spawn_actor(bp_drone, cam_transform)

	return drone

def main():
	static_vehicle_list = []
	dynamic_vehicle_list = []
	dynamic_policy_list = []
	try:
		client = carla.Client("localhost", 2000)
		client.set_timeout(2.0)

		world = client.get_world()
		world = client.load_world("Town05")
		world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

		static_vehicle_list = setup_static_cars(world)
		dynamic_vehicle_list, dynamic_policy_list = setup_dynamic_cars(world)
		drone = setup_camera(world)

		completed = False
		fps = 20
		use_spectator_view = False # if this is enabled, will move the drone transform to the spectator view
		with CarlaSyncMode(world, drone, fps=fps) as sync_mode:
			if use_spectator_view:
				spectator_transform = world.get_spectator().get_transform()
				drone.set_transform(spectator_transform)

			while not completed:
				snap, img = sync_mode.tick(timeout=2.0)

				# Handle drone view.
				img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
				img_array = np.reshape(img_array, (img.height, img.width, 4))
				img_array = img_array[:, :, :3]

				img_array = cv2.resize(img_array, (960, 500), interpolation = cv2.INTER_AREA)
				cv2.imshow('Drone', img_array); cv2.waitKey(1)

				# Handle updating the dynamic cars.  Terminate once all cars reach the goal.
				completed = True
				for act, policy in zip(dynamic_vehicle_list, dynamic_policy_list):
					control = policy.run_step(1./float(fps))
					completed = completed and policy.done()
					act.apply_control(control)

	finally:
		for actor in static_vehicle_list:
			actor.destroy()

		for actor in dynamic_vehicle_list:
			actor.destroy()

		drone.destroy()

		cv2.destroyAllWindows()
		print('Done.')

if __name__ == '__main__':
	main()