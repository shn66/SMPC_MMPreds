import carla
import os
import sys
import cv2
import json
import numpy as np

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario,

def setup_intersection_scenario(scenario_dict, ego_init_dict, savedir):
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
            # Not generating static vehicles
            # vehicles_params_list.append( VehicleParams(**vp_dict) )
            continue
        elif vp_dict["role"] == "target":
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "ego":
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = policy_type
            vp_dict["smpc_config"] = policy_config
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:
            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    runner = RunIntersectionScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    return runner

def get_drone_snapshot(runner):
	img_drone = None
	with CarlaSyncMode(runner.world, runner.drone, fps=runner.carla_fps) as sync_mode:
		_, img = sync_mode.tick(timeout=runner.timeout)
		img_drone = np.frombuffer(img.raw_data, dtype=np.uint8)
        img_drone = np.reshape(img_drone, (img.height, img.width, 4))
        img_drone = img_drone[:, :, :3]
        img_drone = cv2.resize(img_drone, (runner.viz_params.img_width, runner.viz_params.img_height), interpolation = cv2.INTER_AREA)
    return img_drone

def overlay_trajectories(img, runner, radius=2):

	for (veh_policy, veh_color) in zip(runner.vehicle_policies, runner.vehicle_colors):
		veh_color = veh_color[::-1] # RGB to BGR

		if ~veh_policy.hasattr("reference"):
			continue

		xy_traj = veh_policy.reference[:, 1:3]

		for xy in xy_traj:
			px = runner.A_world_to_drone @ xy + runner.b_world_to_drone
			center_x = int(px[0])
			center_y = int(px[1])
			cv2.circle(img, (center_x, center_y), radius, veh_color, thickness=-1)

if __name__ == '__main__':
	# Change each time.
	scenario_num  = 1

	# Loading + Setup.
	scenario_path = os.path.join(scriptdir, f"scenarios/scenarios_{scenario_num:02d}.json")
	ego_init_path = os.path.join(scriptdir, "scenarios/ego_init_01.json")

	scenario_dict = json.load(open(scenario_path, "r"))
	ego_init_dict = json.load(open(ego_init_path, "r"))
    scenario_name = scenario_path.split("/")[-1].split('.json')[0]
    savedir = os.path.join( os.path.abspath(__file__).split("scripts")[0], "results/route_viz/" )

	runner = setup_intersection_scenario(scenario_dict, ego_init_dict, savedir)

	img = get_drone_snapshot(runner)
	overlay_trajectories(img, runner)

	cv2.imwrite(os.path.join(savedir, f"{scenario_name}_route.png"), img)
