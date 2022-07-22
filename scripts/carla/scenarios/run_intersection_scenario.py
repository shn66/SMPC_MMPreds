from dataclasses import dataclass
from typing import List

import carla
import os
import sys

import cv2
import numpy as np
import random
import pickle

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from policies.static_agent import StaticAgent
from policies.smpc_agent import SMPCAgent
from policies.mpc_agent import MPCAgent
from policies.bl_smpc_agent import BLSMPCAgent

from rasterizer.agent_history import AgentHistory
from rasterizer.sem_box_rasterizer import SemBoxRasterizer
from utils.frenet_trajectory_handler import fix_angle
from utils.vehicle_geometry_utils import vehicle_name_to_lf_lr

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from models.deploy_multipath_model import DeployMultiPath

"""
Simulation parameter classes.
"""
@dataclass(frozen=True)
class CarlaParams:
    # Carla world settings + intersection definition.
    map_str               : str # e.g. "Town05"
    weather_str           : str # e.g. "ClearNoon"
    fps                   : int # what fps to run the simulator in synchronous mode
    intersection_csv_loc  : str # file location of the csv defining the intersection

    # Carla client settings.
    ip_addr        : str   = "localhost"
    port           : int   = 2000
    timeout_period : float = 2.0

@dataclass(frozen=True)
class DroneVizParams:
    # Parameters for the "drone": camera used to capture Carla scene.
    # By default, this represents a top down view.
    # XY must be specified, as it varies with the choice of intersection.
    x          : float
    y          : float
    z          : float =  50.
    roll       : float =   0.
    pitch      : float = -90.
    yaw        : float =   0.
    img_width  : int   = 1920
    img_height : int   = 1080
    fov        : int   = 90

    # Parameters for how to handle OpenCV img corresponding to the drone.
    visualize_opencv      : bool = True # show OpenCV window as the simulation is occuring
    save_avi              : bool = True # whether to save OpenCV visualization as a video
    overlay_gmm           : bool = True # whether to show the confidence ellipses for predicted agents.
    overlay_ego_info      : bool = True # add a string with text about ego's state/control.
    overlay_mode_probs    : bool = True # add a string with the mode probabilities.
    overlay_traj_hist     : bool = True # add the trajectory history for each agent

@dataclass(frozen=True)
class VehicleParams:
    # High level vehicle/policy selection.
    role          : str # either "ego" [dynamic, our agent], "static" [nonmoving vehicle], or "target" [dynamic vehicle, other agent]
    vehicle_type  : str # currently use one of {"vehicle.audi.tt", "vehicle.mercedes-benz.coupe"}
    vehicle_color : str # currently use "246, 246, 246" for static, "186, 0, 0" for ego, and "65, 63, 197" for dynamic
    policy_type   : str # {"static", mpc", "smpc", "blsmpc"} -> which control policy to use for this agent

    # Initial state and goal location selection.
    intersection_start_node_idx : int        # {0, 1, 2, 3} -> corresponds to a direction in the intersection_json above
    intersection_goal_node_idx  : int        # {0, 1, 2, 3} -> corresponds to a direction in the intersection_json above
    start_left_offset           : float      # how far to move the car's start pose in its local left (i.e. lateral axis) direction (m)
    goal_left_offset            : float      # how far to move the car's goal pose in its local left (i.e. lateral axis) direction (m)
    start_longitudinal_offset   : float      # how far to move the car's start pose in its local fwd (i.e. longitudinal axis) direction (m)
    goal_longitudinal_offset    : float      # how far to move the car's goal pose in its local fwd (i.e. longitudinal axis) direction (m)
    nominal_speed               : float      # how fast the car should travel if unobstructed / not turning
    init_speed                  : float      # the car's initial speed in simulation (m/s)

    # General MPC parameters.  Some of these can be ignored (e.g. n_modes if using MPCAgent).
    N         : int   = 10  # horizon of the MPC solution
    dt        : float = 0.2 # timestep of the discretization used (s)
    num_modes : int   = 3   # number of GMM modes considered by MPC (prioritizing most probable ones first)

    # SMPC specific parameters (ignored for any other policy_type).
    smpc_config : str = "full" # "full", "open_loop", "no_switch"

@dataclass(frozen=True)
class PredictionParams:
    # Model parameter locations, given relative to <ROOTDIR>/scripts/models/
    model_weights         : str = "l5kit_multipath_10/"
    model_anchors         : str = "l5kit_clusters_16.npy"

    # Flag to render traffic lights on rasterized image for prediction.
    render_traffic_lights : bool = False # set by default to false since agents ignore lights at the moment.

    # TODO: future work includes things like how often to update preds (if not at the Carla fps).

"""
Util functions for Carla. # TODO: move this elsewhere.
"""
def load_intersection(intersection_csv):
    with open(intersection_csv, 'r') as f:
        lines = f.readlines()

    intersection = []

    for line in lines:
        if '#' in line:
            continue # comment
        data = line.replace(" ", "").split(",")
        start_pose = [float(data[0]), float(data[1]), int(data[2])]
        goal_pose  = [float(data[3]), float(data[4]), int(data[5])]
        intersection.append( [start_pose, goal_pose] )

    return intersection

def get_vehicle_policy(vehicle_params, vehicle_actor, goal_transform):
    if vehicle_params.policy_type == "static":
        return StaticAgent(vehicle_actor, goal_transform.location)
    elif vehicle_params.policy_type == "mpc":
        return MPCAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed)
    elif vehicle_params.policy_type == "blsmpc":
        return BLSMPCAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed)
    elif vehicle_params.policy_type == "smpc":
        if vehicle_params.smpc_config.endswith("OAinner"):
            return SMPCAgent(vehicle_actor, goal_transform.location, \
                            N=vehicle_params.N,
                            dt=vehicle_params.dt,
                            N_modes=vehicle_params.num_modes,
                            nominal_speed_mps=vehicle_params.nominal_speed,
                            smpc_config=vehicle_params.smpc_config.split("_OAinner")[0],
                            OAIA=True)
        elif vehicle_params.smpc_config.endswith("obca"):
            return SMPCAgent(vehicle_actor, goal_transform.location, \
                            N=vehicle_params.N,
                            dt=vehicle_params.dt,
                            N_modes=vehicle_params.num_modes,
                            nominal_speed_mps=vehicle_params.nominal_speed,
                            smpc_config=vehicle_params.smpc_config.split("_obca")[0][:-2],
                            obca=True,
                            obca_mode=int(vehicle_params.smpc_config.split("_obca")[0][-1]))
        else :
            return SMPCAgent(vehicle_actor, goal_transform.location, \
                            N=vehicle_params.N,
                            dt=vehicle_params.dt,
                            N_modes=vehicle_params.num_modes,
                            nominal_speed_mps=vehicle_params.nominal_speed,
                            smpc_config=vehicle_params.smpc_config)
    else:
        raise ValueError(f"Unsupported policy type: {vehicle_params.policy_type}")

def get_intersection_transform(intersection, vehicle_params, endpoint_str, spawn_height=1.0):
    node_idx            = None
    endpoint_idx        = None
    left_offset         = None
    longitudinal_offset = None

    if endpoint_str == "start":
        endpoint_idx        = 0
        node_idx            = vehicle_params.intersection_start_node_idx
        left_offset         = vehicle_params.start_left_offset
        longitudinal_offset = vehicle_params.start_longitudinal_offset

    elif endpoint_str == "goal":
        endpoint_idx        = 1
        node_idx            = vehicle_params.intersection_goal_node_idx
        left_offset         = vehicle_params.goal_left_offset
        longitudinal_offset = vehicle_params.goal_longitudinal_offset
    else:
        raise ValueError(f"Invalid endpoint:{endpoint_str}.  Expected start or goal.")

    # Extract the pose from the intersection definition.
    x, y, yaw_deg = intersection[node_idx][endpoint_idx]
    yaw_rad = np.radians(float(yaw_deg))

    # Translate the pose given the longitudinal offset.
    x += longitudinal_offset * np.cos(yaw_rad)
    y += longitudinal_offset * np.sin(yaw_rad)

    # Translate the pose given the lateral/left offset.
    left_dir_yaw = yaw_rad - np.pi/2.
    x += left_offset * np.cos( left_dir_yaw )
    y += left_offset * np.sin( left_dir_yaw )

    # Make the Carla transform.
    loc = carla.Location(x = x, y = y, z = spawn_height)
    rot = carla.Rotation(yaw=yaw_deg)
    return carla.Transform(loc, rot)

def transform_to_local_frame(motion_hist_array):
    # TODO: clean up / document / move.
    local_x, local_y, local_yaw = motion_hist_array[-1, 1:]

    R_local_to_world    = np.array([[np.cos(local_yaw), -np.sin(local_yaw)],\
                                    [np.sin(local_yaw),  np.cos(local_yaw)]])
    t_local_to_world    = np.array([local_x, local_y])

    R_world_to_local =  R_local_to_world.T
    t_world_to_local = -R_local_to_world.T @ t_local_to_world

    for t in range(motion_hist_array.shape[0]):
        xy_global = motion_hist_array[t, 1:3]
        xy_local  = R_world_to_local @ xy_global + t_world_to_local
        motion_hist_array[t, 1:3] = xy_local

        pose_diff = motion_hist_array[t, 3] - local_yaw

        if ~np.isnan(pose_diff):
            motion_hist_array[t, 3] = fix_angle(pose_diff)

    return motion_hist_array, R_local_to_world, t_local_to_world

def get_target_agent_history(agent_history, target_agent_id):
    # TODO: clean up / document / move.
    snapshot = agent_history.query(history_secs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    tms   = []
    poses = []

    for k in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        tms.append(k)
        snapshot_key = np.round(k, 2)
        if(len(snapshot[snapshot_key]) == 0):
            poses.append([None, None, None])
        else:
            for entry in snapshot[snapshot_key]['vehicles']:
                if entry['id'] == target_agent_id:
                    pose = entry['centroid']
                    pose.append(entry['yaw'])
                    poses.append( pose )
                    break
    tms = [-v if v > 0. else 0. for v in tms]
    motion_hist_array = np.column_stack((tms, poses)).astype(np.float32)

    return transform_to_local_frame(motion_hist_array)

"""
Main class to simulate and run parametrized scenarios.
"""
class RunIntersectionScenario:
    def __init__(self,
                 carla_params        : CarlaParams,
                 drone_viz_params    : DroneVizParams,
                 vehicle_params_list : List[VehicleParams],
                 prediction_params   : PredictionParams,
                 savedir : str):
        try:
            self._setup_carla_world(carla_params)
            self._setup_vehicles(vehicle_params_list, carla_params)
            self._setup_camera(drone_viz_params)
            self._setup_predictions(prediction_params)
        except Exception as e:
            print("Failed to setup the scenario!")
            raise e

        # For logging results + videos.
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)

        # Needed for Sync mode loop.
        self.timeout   = carla_params.timeout_period
        self.carla_fps = carla_params.fps
        self.max_iters = self.carla_fps*30 # limit scenario run to 30 seconds max

        # Needed for OpenCV/Carla world visualization.
        self.viz_params = drone_viz_params
        self.mode_rgb_colors = [(255, 0, 255), (255, 255, 0), (0, 255, 255)] # TODO: autogenerate

    def run_scenario(self):
        # Return flag to indicate if this ran to completion.
        ran_successfully = False

        # Video Setup
        writer = None
        if self.viz_params.save_avi:
            avi_name = os.path.join(self.savedir, "carla_sim.avi")
            writer   = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc(*'MJPG'), self.carla_fps, (self.viz_params.img_width, self.viz_params.img_height))

        # Data Logging Setup
        self.results_dict = {}
        for ind_vehicle, vehicle in enumerate(self.vehicle_actors):
            key = f"{vehicle.attributes['role_name']}_{ind_vehicle}"
            l_f, l_r = vehicle_name_to_lf_lr(vehicle.type_id) # e.g. "vehicle.audi.tt"
            self.results_dict[key] = {"l_f"              : l_f,
                                      "l_r"              : l_r,
                                      "state_trajectory" : [],
                                      "input_trajectory" : [],
                                      "feasibility"      : [],
                                      "solve_times"      : []}

        try:
            with CarlaSyncMode(self.world, self.drone, fps=self.carla_fps) as sync_mode:
                # Run simulation a couple steps to allow the initial velocities to be processed.

                # Set initial velocity for all vehicle agents.
                for veh_actor, init_speed in zip(self.vehicle_actors, self.vehicle_init_speeds):
                    yaw_carla = veh_actor.get_transform().rotation.yaw
                    carla_vel = carla.Vector3D(x=init_speed*np.cos(np.radians(yaw_carla)) ,
                                               y=init_speed*np.sin(np.radians(yaw_carla)) ,
                                               z=0.)
                    veh_actor.set_velocity(carla_vel)

                for _ in range(2):
                    sync_mode.tick(timeout=self.timeout)

                # Loop until all vehicles have reached their goal or we've exceeded self.max_iters.
                for _ in range(self.max_iters):
                    snap, img = sync_mode.tick(timeout=self.timeout)

                    # Handle predictions.
                    self.agent_history.update(snap, self.world)
                    tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred = self._make_predictions()
                    pred_dict={ "tvs_positions": tvs_positions, "tvs_mode_dists": tvs_mode_dists}

                    # Run policies for each agent.
                    t_elapsed = snap.elapsed_seconds
                    completed = True
                    ego_feas=False

                    for idx_act, (act, policy) in enumerate(zip(self.vehicle_actors, self.vehicle_policies)):
                        control, z0, u0, is_feasible, solve_time = policy.run_step(pred_dict)
                        if not policy.done():
                            z0 = np.append(t_elapsed, z0) # add the Carla timestamp
                            act_key = f"{act.attributes['role_name']}_{idx_act}"
                            self.results_dict[act_key]["state_trajectory"].append(z0)
                            self.results_dict[act_key]["input_trajectory"].append(u0)
                            self.results_dict[act_key]["feasibility"].append(is_feasible)
                            self.results_dict[act_key]["solve_times"].append(solve_time)

                        # true at the end of the loop only if all agents are done or if iter_ctr>=max_iters
                        completed = completed and policy.done()
                        act.apply_control(control)

                        if idx_act == self.ego_vehicle_idx:
                            # Keep track of ego's information for rendering.
                            ego_vel   = act.get_velocity()
                            ego_speed = np.linalg.norm([ego_vel.x, ego_vel.y])
                            ego_ctrl  = control
                            ego_feas=is_feasible

                    # Get drone camera image.
                    img_drone = np.frombuffer(img.raw_data, dtype=np.uint8)
                    img_drone = np.reshape(img_drone, (img.height, img.width, 4))
                    img_drone = img_drone[:, :, :3]
                    img_drone = cv2.resize(img_drone, (self.viz_params.img_width, self.viz_params.img_height), interpolation = cv2.INTER_AREA)

                    # Handle overlays on drone camera image.
                    if self.viz_params.overlay_ego_info:
                        ego_str = f"EGO - v:{ego_speed:.3f}, th: {ego_ctrl.throttle:.2f}, bk: {ego_ctrl.brake:.2f}, st: {ego_ctrl.steer:.2f}"
                        cv2.putText(img_drone, ego_str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if not ego_feas:
                        feas_str = "INFEASIBLE"
                        cv2.putText(img_drone, feas_str, (100,600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)


                    if self.viz_params.overlay_gmm:
                        if tvs_valid_pred[0]: # TODO: generalize this to multiple TVs.
                            self._viz_gmm(img_drone, tvs_mode_dists)

                    if self.viz_params.overlay_traj_hist:
                        self._viz_traj_hist(img_drone)

                    if self.viz_params.overlay_mode_probs:
                        if tvs_valid_pred[0]: # TODO: generalize this to multiple TVs.
                            cv2.putText(img_drone, "Mode probabilities: ", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            for prob_idx, mode_prob in enumerate(tvs_mode_probs[0]):
                                cv2.putText(img_drone, f"{mode_prob:.3f}",
                                            (360 + prob_idx * 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.mode_rgb_colors[prob_idx], 2)


                    # Handle visualization / saving to video.
                    if self.viz_params.visualize_opencv:
                        cv2.imshow("Drone", img_drone); cv2.waitKey(1)

                    if self.viz_params.save_avi:
                        writer.write(img_drone)

                    if completed:
                        # All cars reached their destinations, end before self.max_iters.
                        break

                # Save results and mark successful completion.
                for act_key in self.results_dict:
                    for arr_key in ["state_trajectory",
                                    "input_trajectory",
                                    "feasibility",
                                    "solve_times"]:
                        self.results_dict[act_key][arr_key] = np.array(self.results_dict[act_key][arr_key])
                pkl_name = os.path.join(self.savedir, "scenario_result.pkl")
                pickle.dump(self.results_dict, open(pkl_name, "wb"))
                ran_successfully = True

        # Teardown.
        finally:
            if writer:
                writer.release()
            for actor in self.vehicle_actors:
                actor.destroy()
            self.drone.destroy()
            cv2.destroyAllWindows()

        return ran_successfully

    def _make_predictions(self):
        if len(self.tv_vehicle_idxs) == 0:
            ego_location = self.vehicle_actors[self.ego_vehicle_idx].get_location()
            ego_x, ego_y = ego_location.x, -ego_location.y
            curr_target_vehicle_position = [1000 + ego_x, 1000 + ego_y]
            tvs_positions = [curr_target_vehicle_position]
            tvs_mode_probs = [ np.ones(self.ego_num_modes) / self.ego_num_modes ]
            tvs_mode_dists = [[np.stack([[curr_target_vehicle_position]*self.ego_N]*self.ego_num_modes)],
                              [np.stack([[np.identity(2)]*self.ego_N]*self.ego_num_modes)]]
            tvs_valid_pred = [False]
        else:
            # TODO: clean up and generalize this to many target vehicles.
            target_agent_id = self.vehicle_actors[self.tv_vehicle_idxs[0]].id
            past_states_tv, R_target_to_world, t_target_to_world = \
                get_target_agent_history(self.agent_history, target_agent_id)

            curr_target_vehicle_position = R_target_to_world @ past_states_tv[-1, 1:3] + t_target_to_world
            tvs_positions = [curr_target_vehicle_position]

            if np.any(np.isnan(past_states_tv)):
                # Not enough data for predictions to be made.
                tvs_mode_probs = [ np.ones(self.ego_num_modes) / self.ego_num_modes ]
                tvs_mode_dists = [[np.stack([[curr_target_vehicle_position]*self.ego_N]*self.ego_num_modes)],
                                  [np.stack([[0.1*np.identity(2)]*self.ego_N]*self.ego_num_modes)]]
                tvs_valid_pred = [False]
            else:
                img_tv = self.rasterizer.rasterize(self.agent_history, target_agent_id)
                gmm_pred_tv = self.pred_model.predict_instance(img_tv, past_states_tv[:-1])
                gmm_pred_tv.transform(R_target_to_world, t_target_to_world)
                gmm_pred_tv=gmm_pred_tv.get_top_k_GMM(self.ego_num_modes+9)

                tvs_mode_probs = [gmm_pred_tv.mode_probabilities[::4]]
                tvs_mode_dists = [[gmm_pred_tv.mus[::2, :self.ego_N, :]], [gmm_pred_tv.sigmas[::2, :self.ego_N, :, :]]]
                tvs_valid_pred = [True]

        return tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred

    def _setup_carla_world(self, carla_params):
        client = carla.Client(carla_params.ip_addr, carla_params.port)
        client.set_timeout(carla_params.timeout_period)
        self.world = client.load_world(carla_params.map_str)
        self.world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

    def _setup_camera(self, drone_viz_params):
        bp_library = self.world.get_blueprint_library()
        bp_drone  = bp_library.find('sensor.camera.rgb')

        # TODO: compute these, hardcoded for now.
        self.A_world_to_drone = np.array([[    0., -19.2],
                                          [-19.2,      0.]])
        self.b_world_to_drone = np.array([ 960., 1116.])

        # This is like a top down view of the intersection.  Can tune later.
        cam_loc = carla.Location(x=drone_viz_params.x,
                                 y=drone_viz_params.y,
                                 z=drone_viz_params.z)
        cam_ori = carla.Rotation(roll=drone_viz_params.roll,
                                 pitch=drone_viz_params.pitch,
                                 yaw=drone_viz_params.yaw)
        cam_transform = carla.Transform(cam_loc, cam_ori)

        bp_drone.set_attribute('image_size_x', str(drone_viz_params.img_width))
        bp_drone.set_attribute('image_size_y', str(drone_viz_params.img_height))
        bp_drone.set_attribute('fov', str(drone_viz_params.fov))
        bp_drone.set_attribute('role_name', 'drone')

        self.drone = self.world.spawn_actor(bp_drone, cam_transform)

    def _setup_vehicles(self, vehicle_params_list, carla_params):
        intersection_fname = os.path.join( os.path.dirname(os.path.abspath(__file__)),
                                           carla_params.intersection_csv_loc )
        intersection = load_intersection(intersection_fname)
        bp_library = self.world.get_blueprint_library()

        self.vehicle_actors   = []
        self.vehicle_policies = []
        self.vehicle_colors   = []
        self.vehicle_init_speeds = []
        ego_vehicle_idxs  = []
        tv_vehicle_idxs   = []

        for idx, vp in enumerate(vehicle_params_list):
            veh_bp = bp_library.find(vp.vehicle_type)
            veh_bp.set_attribute("color", vp.vehicle_color)
            veh_bp.set_attribute("role_name", vp.role)
            self.vehicle_colors.append([int(x) for x in vp.vehicle_color.split(", ")])

            if vp.role == "ego":
                ego_vehicle_idxs.append(idx)
            elif vp.role == "static":
                pass
            elif vp.role == "target":
                tv_vehicle_idxs.append(idx)
            else:
                raise ValueError(f"Invalid vehicle role selection : {vp.role}")

            start_transform = get_intersection_transform(intersection, vp, "start")
            goal_transform  = get_intersection_transform(intersection, vp, "goal")

            veh_actor  = self.world.spawn_actor(veh_bp, start_transform)
            veh_policy = get_vehicle_policy(vp, veh_actor, goal_transform)

            self.vehicle_actors.append(veh_actor)
            self.vehicle_policies.append(veh_policy)
            self.vehicle_init_speeds.append(vp.init_speed)

        if len(ego_vehicle_idxs) != 1:
            raise RuntimeError(f"Invalid number of ego vehicles spawned: {len(ego_vehicle_idxs)}")
        self.ego_vehicle_idx = ego_vehicle_idxs[0]
        self.ego_N           = vehicle_params_list[self.ego_vehicle_idx].N
        self.ego_num_modes   = vehicle_params_list[self.ego_vehicle_idx].num_modes

        # Note: this can be empty, as checked in the _make_predictions code.
        self.tv_vehicle_idxs = tv_vehicle_idxs

    def _setup_predictions(self, prediction_params):
        self.agent_history = AgentHistory(self.world.get_actors())
        self.rasterizer    = SemBoxRasterizer(self.world.get_map().get_topology(), render_traffic_lights=\
                                                 prediction_params.render_traffic_lights)
        prefix             = os.path.abspath(__file__).split('carla')[0] + 'models/'
        self.pred_model    = DeployMultiPath(prefix+prediction_params.model_weights, \
                                             np.load(prefix+prediction_params.model_anchors))

        # Try to do a sample prediction, initialize and check GPU model is working fine.
        blank_image = np.zeros((self.rasterizer.sem_rast.raster_height,
                                self.rasterizer.sem_rast.raster_width,
                                3), dtype=np.uint8)
        zero_traj   = np.column_stack(( np.arange(-1.0, 0.00, 0.2),
                                        np.zeros((5,3))
                                      )).astype(np.float32)
        self.pred_model.predict_instance(image_raw   = blank_image,
                                         past_states = zero_traj)

    def _viz_gmm(self, img, tvs_mode_dists, mdist_sq_thresh=5.991):
        mus    = tvs_mode_dists[0][0] # N_modes by N by 2
        sigmas = tvs_mode_dists[1][0] # N_modes by N by 2 by 2

        # Note: we reverse mode_dists and colors s.t. least probable mode is plotted first.
        # zip_obj = zip( reversed(mus), reversed(sigmas), reversed(self.mode_rgb_colors) )
        zip_obj = zip( mus, sigmas, self.mode_rgb_colors )

        for mean_traj, covar_traj, color in zip_obj:
            color = color[::-1] # rgb to bgr
            for (mean_xy, covar_xy) in zip(mean_traj, covar_traj):
                mu_px    = self.A_world_to_drone @ mean_xy + self.b_world_to_drone
                center_x = int(mu_px[0])
                center_y = int(mu_px[1])
                covar_px = self.A_world_to_drone @ covar_xy @ self.A_world_to_drone.T
                evals_px, evecs_px = np.linalg.eigh(covar_px)

                length_ax1 = int( np.sqrt(mdist_sq_thresh * evals_px[0]) ) # half the first axis diameter in pixels
                length_ax2 = int( np.sqrt(mdist_sq_thresh * evals_px[1]) ) # half the second axis diameter in pixels
                ang_1 = -np.degrees( np.arctan2(evecs_px[1,0], evecs_px[0,0]) ) # -ang since cv2.ellipse uses clockwise angle
                cv2.ellipse( img, (center_x, center_y), (length_ax1, length_ax2), ang_1, 0, 360, color, thickness=2)

    def _viz_traj_hist(self, img, radius=2):
        for idx_act, (act, act_color) in enumerate(zip(self.vehicle_actors,self.vehicle_colors)):
            act_key = f"{act.attributes['role_name']}_{idx_act}"
            act_traj = self.results_dict[act_key]["state_trajectory"]
            act_color = act_color[::-1] # rgb to bgr

            for act_st in act_traj:
                act_xy = np.array(act_st[1:3])
                act_px = self.A_world_to_drone @ act_xy + self.b_world_to_drone
                center_x = int(act_px[0])
                center_y = int(act_px[1])
                cv2.circle(img, (center_x, center_y), radius, act_color, thickness=-1)