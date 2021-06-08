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

"""
Start and End Locations for 2 intersections in Town05.
Just getting the outer lanes, this list is not comprehensive.
TODO: put in a json.

[X_carla, Y_carla, yaw_carla].
"""
INTERSECTIONS = []

# First intersection
INTERSECTIONS.append([
[[14.5, 6.0, 0], [43.1, 5.1, 0]],
[[24.4, -16.3, 90], [24.6, 14.7, 90]],
[[44.4, -5.2, 180], [15.2, -4.7, 180]],
[[35.0, 15.0, 270], [35.0, -14.6, 270]]
])

# Second intersection
INTERSECTIONS.append([
[[87.3, 5.5, 0], [115.0, 1.6, 0]],
[[96.3, -15.2, 90], [96.1, 14.1, 90]],
[[115.5, -2.2, 180], [88.1, -5.1, 180]],
[[106.7, 14.4, 270], [107.0, -13.7, 270]]
])

INT_IND = 0 # which intersection to use, 0 or 1 atm.
LAN_IND = 3 # which straight-line lane to consider, basically heading.

EGO_START_LOCATION = (*INTERSECTIONS[INT_IND][LAN_IND][0][:-1], 1.5)
EGO_END_LOCATION   = (*INTERSECTIONS[INT_IND][LAN_IND][1][:-1], 1.5)
EGO_YAW            = INTERSECTIONS[INT_IND][LAN_IND][0][-1]

TOWN_STR = "Town05"
HOST = "127.0.0.1"
PORT = 2000
EGO_VEH_FILTER = "vehicle.lincoln.mkz2017"

try:
	client = carla.Client(HOST, PORT)
	client.set_timeout(2.0)
	world = client.get_world()
	# world = client.load_world(TOWN_STR)
	world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))


	bp_library = world.get_blueprint_library()

	""" EGO vehicle """
	ego_location = carla.Location(*EGO_START_LOCATION)
	ego_rotation = carla.Rotation(yaw=EGO_YAW)
	ego_transform = carla.Transform(ego_location, ego_rotation)
	ego_vehicle = random.choice(bp_library.filter(EGO_VEH_FILTER))
	ego_vehicle.set_attribute('role_name', 'hero')
	ego_vehicle = world.spawn_actor(ego_vehicle, ego_transform)

	time.sleep(5.)

	ego_vehicle.set_transform( carla.Transform(carla.Location(*EGO_END_LOCATION), \
		                       ego_rotation) )
	time.sleep(5.)

finally:
	ego_vehicle.destroy()
	print('Done.')
