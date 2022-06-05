import glob
import os
import sys
import time

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
	client.set_timeout(2.0)
	# world = client.get_world()
	world = client.load_world("Town04")

	# print(help(t))
	# print("(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z))


	while(True):
		t = world.get_spectator().get_transform()
		# coordinate_str = "(x,y) = ({},{})".format(t.location.x, t.location.y)
		coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
		print (coordinate_str)
		time.sleep(_SLEEP_TIME_)



if __name__ == '__main__':
	main()