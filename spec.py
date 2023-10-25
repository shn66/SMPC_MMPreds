import glob
import os
import sys
import time
import random

CARLA_ROOT = os.getenv("CARLA_ROOT")
try:
    sys.path.append(glob.glob(CARLA_ROOT+'./PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

_HOST_ = 'localhost'
_PORT_ = 2000
_SLEEP_TIME_ = 2


def main():
	client = carla.Client(_HOST_, _PORT_)
	client.set_timeout(999)
	# world = client.get_world()
	world = client.load_world("Town04")

	# print(help(t))
	# print("(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z))
	blueprint_library = world.get_blueprint_library()
	bp = random.choice(blueprint_library.filter('vehicle.bmw.grandtourer'))

	transform = random.choice(world.get_map().get_spawn_points())
	transform.location.x = -328.4
	transform.location.y = 36.7
	transform.location.z = 5
	transform.rotation.yaw = 0
	transform.rotation.pitch = 0
	transform.rotation.roll = 0
	vehicle = world.spawn_actor(bp, transform)

	spec = world.get_spectator()
	
	spec.set_transform(transform)

	transform1=transform
	transform1.location.x = -315.
	transform1.location.y = 36.7
	transform1.location.z = 5
	transform1.rotation.yaw = 0
	transform1.rotation.pitch = 0
	transform1.rotation.roll = 0
	vehicle1 = world.spawn_actor(bp, transform1)

	transform2=transform
	transform2.location.x = -354.6
	transform2.location.y = 33.8
	transform2.location.z = 5
	transform2.rotation.yaw = 0
	transform2.rotation.pitch = 0
	transform2.rotation.roll = 0
	vehicle2 = world.spawn_actor(bp, transform2)

	while(True):
		t = world.get_spectator().get_transform()
		

		# coordinate_str = "(x,y) = ({},{})".format(t.location.x, t.location.y)
		coordinate_str = "(x,y,z) = ({},{},{},{})".format(t.location.x, t.location.y,t.location.z, t.rotation.yaw)
		print (coordinate_str)
		time.sleep(_SLEEP_TIME_)



if __name__ == '__main__':
	main()