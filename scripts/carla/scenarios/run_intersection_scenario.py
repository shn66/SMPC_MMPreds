from dataclasses import dataclass

@dataclass(frozen=True)
class CarlaParams:
	# Carla client settings.
	carla_ip_addr  : str   = "localhost"
	carla_port     : int   = 2000
	timeout_period : float = 2.0

	# Carla world settings + intersection definition.
	map_str               : str # e.g. "Town05"
	weather_str           : str # e.g. "ClearNoon"
	carla_sync_fps        : int # what fps to run the simulator in synchronous mode
	intersection_json_loc : str # file location of the json defining the intersection

@dataclass(frozen=True)
class DroneVizParams:
	# Parameters for the "drone": camera used to capture Carla scene.
	# By default, this represents a top down view.
	# XY must be specified, as it varies with the choice of intersection.
	drone_x          : float
	drone_y          : float
	drone_z          : float =  50.
	drone_roll       : float =   0.
	drone_pitch      : float = -90.
	drone_yaw        : float =   0.
	drone_img_width  : int   = 1920
	drone_img_height : int   = 1080
	drone_fov        : int   = 90

	# Parameters for how to handle OpenCV img corresponding to the drone.
	visualize_opencv      : bool = true # show OpenCV window as the simulation is occuring
	save_avi              : bool = true # whether to save OpenCV visualization as a video
	overlay_gmm           : bool = true # whether to show the confidence ellipses for predicted agents.
	overlay_ego_info      : bool = true # Add a string with text about ego's state/control.
	overlay_mode_probs    : bool = true # Add a string with the mode probabilities.

@dataclass(frozen=True)
class VehicleParams:
	# High level vehicle/policy selection.
	role          : str # either "ego" [dynamic, our agent], "static" [nonmoving vehicle], or "target" [dynamic vehicle, other agent]
	vehicle_type  : str # currently use one of {"vehicle.audi.tt", "vehicle.mercedes-benz.coupe"}
	vehicle_color : str # currently use "246, 246, 246" for static, "186, 0, 0" for ego, and "65, 63, 197" for dynamic
	policy_type   : str # {"mpc", "smpc", "smpc_bl"} -> which contorl policy to use for this agent

	# Initial state and goal location selection.
	intersection_start_node_idx : int        # {0, 1, 2, 3} -> corresponds to a direction in the intersection_json above
	intersection_goal_node_idx  : int        # {0, 1, 2, 3} -> corresponds to a direction in the intersection_json above
	start_lane_selection        : str        # {"left", "right"}  -> which lane in that starting road segment to be in
	goal_lane_selection         : str        # {"left", "right"}  -> which lane in that ending road segment to be in
	start_longitudinal_offset   : float      # how far to move the car's start location along the lane (m)
	goal_longitudinal_offset    : float      # how far to move the car's goal location along the lane (m)
	init_speed                  : float = 0. # the car's initial speed in simulation (m/s)

	# General MPC parameters.  Some of these can be ignored (e.g. n_modes if using MPCAgent).
	N       : int   = 10  # horizon of the MPC solution
	dt      : float = 0.2 # timestep of the discretization used (s)
	n_modes : int   = 3   # number of GMM modes considered by MPC (prioritizing most probable ones first)

	# SMPC specific parameters (ignored for any other policy_type).
	smpc_config : str # "full", "open_loop", "no_switch"

@dataclass(frozen=True)
class PredictionParams:
	# Model parameter locations, given relative to <ROOTDIR>/scripts/models/
	model_weights         : str = "l5kit_multipath_10/"
	model_anchors         : str = "l5kit_clusters_16.npy"

	# Flag to render traffic lights on rasterized image for prediction.
	render_traffic_lights : bool = false # set by default to false since agents ignore lights at the moment.

	# TODO: future work includes things like how often to update preds (if not at the carla_sync_fps).

class RunIntersectionScenario:
	def __init__(self,
		         carla_params        : CarlaParams,
		         drone_params        : DroneVizParams,
		         vehicle_params_list : List[VehicleParams],
		         prediction_params   : PredictionParams,
		         save_location : str):
		"""
		TODO
		(1) setup Carla world
		(2) setup drone + OpenCV stuff
		(3) setup predictions
		(4) setup all vehicles, set initial state
		(5) save relevant flags / objects needed to run scenario from params
		"""
		pass

	def run_scenario(self):
		"""
		TODO
		(1) setup CarlaSyncMode
		(2) initialize vehicle speed if not already done
		(3) rollout policies which output (control, state, input) and log trajectories; stop when all agents reach goal.
		(4) in the process, show the video (if asked) and write frames to avi (if asked)
		(5) do teardown on completion or error -> save results, close all OpenCV stuff, destroy all actors
		"""
		pass