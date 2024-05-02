import os
import glob
import json
import pdb




def run_without_tvs(scene, scenario_dict, ego_init_dict, savedir, get_cl=False):
    if scene =="intersection":
        from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario
    else:
        from scenarios.run_lk_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunLKScenario


    carla_params     = CarlaParams(**scenario_dict["carla_params"])
    drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
    pred_params      = PredictionParams()

    vehicles_params_list = []

    for vp_dict in scenario_dict["vehicle_params"]:
        if vp_dict["role"] == "static":
            continue
            # vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif "target" in vp_dict["role"]:
            pass
        elif vp_dict["role"] == "ego":
            if get_cl:
                vp_dict['goal_left_offset']=0.0
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = "blsmpc"
            vp_dict["smpc_config"] = ""
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:

            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    if scene =="intersection":
        runner = RunIntersectionScenario(carla_params,
                                        drone_viz_params,
                                        vehicles_params_list,
                                        pred_params,
                                        savedir)
    else:
        runner = RunLKScenario(carla_params,
                                        drone_viz_params,
                                        vehicles_params_list,
                                        pred_params,
                                        savedir)
    
    runner.run_scenario()

def run_with_tvs(scene, scenario_dict, ego_init_dict, ego_policy_config, savedir):
    if scene =="intersection":
        from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario
    else:
        from scenarios.run_lk_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunLKScenario
    
    
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
    elif ego_policy_config == "mpc":
        policy_type = "mpc"
        policy_config = ""
    else:
        raise ValueError(f"Invalid ego policy config: {ego_policy_config}")

    for vp_dict in scenario_dict["vehicle_params"]:
        if vp_dict["role"] == "static":
            # Not generating static vehicles
            vehicles_params_list.append( VehicleParams(**vp_dict) )
            # continue
        elif "target" in vp_dict["role"]:
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "ego":
         
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = policy_type
            vp_dict["smpc_config"] = policy_config
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:

            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    if scene == "intersection":
        runner = RunIntersectionScenario(carla_params,
                                        drone_viz_params,
                                        vehicles_params_list,
                                        pred_params,
                                        savedir)
    else:
        runner = RunLKScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    runner.run_scenario()

if __name__ == '__main__':
    scenario_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/" )
    scenarios_list = sorted(glob.glob(scenario_folder + "scenario_lk.json"))
    results_folder = os.path.join( os.path.abspath(__file__).split("scripts")[0], "results" )

    for scenario in scenarios_list:
        # Load the scenario and generate parameters.
        scenario_dict = json.load(open(scenario, "r"))
        scenario_name = scenario.split("/")[-1].split('.json')[0]
        if "lk" in scenario_name:
            scene = "highway"
        else:
            scene = "intersection"
        inits_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/inits/" )

        

        ego_init_list = sorted(glob.glob(inits_folder + "ego_init_01.json"))

        for ego_init in ego_init_list:
            # Load the ego vehicle parameters.
            ego_init_dict = json.load(open(ego_init, "r"))
            ego_init_name = ego_init.split("/")[-1].split(".json")



            # Run first without any target vehicles.
            # savedir = os.path.join( results_folder, f"{scenario_name}_{ego_init_name}_notv")
            # run_without_tvs(scene, scenario_dict, ego_init_dict, savedir)

            # # Get centerline
            # savedir = os.path.join( results_folder, f"{scenario_name}_{ego_init_name}_notv_cl")
            # run_without_tvs(scene, scenario_dict, ego_init_dict, savedir, get_cl=True)
            # print("****************\n"
            #       f"{scenario_name}_{ego_init_name}_notv\n"
            #       "****************\n")
            
            # Run all ego policy options with target vehicles.
            
         
            for ego_policy_config in ["smpc_var_risk", "smpc_open_loop", "smpc_fixed_risk"]:
      

              savedir = os.path.join( results_folder,
                                      f"{scenario_name}_{ego_init_name}_{ego_policy_config}")
              print("****************\n"
                  f"{scenario_name}_{ego_init_name}_{ego_policy_config}\n"
                  "****************\n")
              run_with_tvs(scene, scenario_dict, ego_init_dict, ego_policy_config, savedir)

            # break
