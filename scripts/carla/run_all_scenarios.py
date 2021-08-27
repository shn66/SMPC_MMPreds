import os
import glob
import json
import pdb

from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario

def run_without_tvs(scenario_dict, ego_init_dict):
	carla_params     = CarlaParams(**scenario_dict["carla_params"])
	drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
	pred_params      = PredictionParams()

	vehicles_params_list = []

	for vp_dict in scenario_dict["vehicle_params"]:
		if vp_dict["role"] == "static":
			vehicles_params_list.append( VehicleParams(**vp_dict) )
		elif vp_dict["role"] == "target":
			pass
		elif vp_dict["role"] == "ego":
			vp_dict.update(ego_init_dict)
			vp_dict["policy_type"] = "smpc"
			vp_dict["smpc_config"] = "open_loop"
			vehicles_params_list.append( VehicleParams(**vp_dict) )
		else:
			raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

	import pdb; pdb.set_trace()
	# runner = RunIntersectionScenario(carla_params,
	# 	                             drone_viz_params,
	# 	                             vehicles_params_list,
	# 	                             pred_params,
	# 	                             "") # TODO
	# runner.run_scenario()


def run_with_tvs(scenario_dict, ego_init_dict, ego_policy_config):
	carla_params     = CarlaParams(**scenario_dict["carla_params"])
	drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
	pred_params      = PredictionParams()

	vehicles_params_list = []

	if ego_policy_config == "blsmpc":
		policy_type   = "blsmpc"
		policy_config = ""
	elif ego_policy_config.startswith("smpc"):
		policy_type = "smpc"
		policy_config = ego_policy_config.split("smpc_")[-1]
	else:
		raise ValueError(f"Invalid ego policy config: {ego_policy_config}")

	for vp_dict in scenario_dict["vehicle_params"]:
		if vp_dict["role"] == "static":
			vehicles_params_list.append( VehicleParams(**vp_dict) )
		elif vp_dict["role"] == "target":
			vehicles_params_list.append( VehicleParams(**vp_dict) )
		elif vp_dict["role"] == "ego":
			vp_dict.update(ego_init_dict)
			vp_dict["policy_type"] = policy_type
			vp_dict["smpc_config"] = policy_config
			vehicles_params_list.append( VehicleParams(**vp_dict) )
		else:
			raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

	import pdb; pdb.set_trace()
	# runner = RunIntersectionScenario(carla_params,
	# 	                             drone_viz_params,
	# 	                             vehicles_params_list,
	# 	                             pred_params,
	# 	                             "") # TODO
	# runner.run_scenario()

if __name__ == '__main__':
	scenario_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/" )
	scenarios_list = glob.glob(scenario_folder + "scenario_*.json")

	for scenario in scenarios_list:
		# Load the scenario and generate parameters.
		scenario_dict = json.load(open(scenario, "r"))
		scenario_name = scenario.split("/")[-1]

		ego_init_list = scenario["ego_init_jsons"]
		for ego_init in ego_init_list:
			# Load the ego vehicle parameters.
			ego_init_dict = json.load(open(os.path.join(scenario_folder, ego_init), "r"))

			# Run first without any target vehicles.
			run_without_tvs(scenario_dict, ego_init_dict)

			# Run all ego policy options with target vehicles.
			for ego_policy_config in ["blsmpc", "smpc_full", "smpc_open_loop", "smpc_no_switch"]:
				run_with_tvs(scenario_dict, ego_init_dict, ego_policy_config)
