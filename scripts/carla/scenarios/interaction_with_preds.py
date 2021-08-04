import carla
import os
import sys

import cv2
import numpy as np
import random

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from policies.frenet_pid_agent import FrenetPIDAgent
from policies.smpc_agent import SMPCAgent
from policies.mpc_agent import MPCAgent
from policies.bl_smpc_agent import BLMPCAgent

from rasterizer.agent_history import AgentHistory
from rasterizer.sem_box_rasterizer import SemBoxRasterizer
from utils.frenet_trajectory_handler import fix_angle

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from models.deploy_multipath_model import DeployMultiPath

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

SAVEDMODELH5 = os.path.abspath(__file__).split('carla')[0] + 'models/l5kit_multipath_10/'
ANCHORS      = np.load(os.path.abspath(__file__).split('carla')[0] + 'models/l5kit_clusters_16.npy')

SCENARIO_CASE = 2
DYNAMIC_CARS = []
if SCENARIO_CASE == 0:
    DYNAMIC_CARS  = [[[0,0,'L'], [3,1,'L'], SMPCAgent],       # facing east, turn left towards north
                     [[2,0,'L'], [2,1,'L'], MPCAgent]]  # oncoming driving west
elif SCENARIO_CASE == 1:
    DYNAMIC_CARS  = [[[0,0,'R'], [1,1,'R'], SMPCAgent],       # facing east, turn right towards south
                     [[2,0,'L'], [1,1,'L'], MPCAgent]]  # facing west, turning left towards south
elif SCENARIO_CASE == 2:
    DYNAMIC_CARS  = [[[0,0,'L'], [3,1,'L'], SMPCAgent],       # facing east, turn left towards north
                     [[2,0,'L'], [1,1,'L'], MPCAgent]]  # facing west, turning left towards south
elif SCENARIO_CASE == 3:
    DYNAMIC_CARS  = [[[0,0,'L'], [0,1,'L'], SMPCAgent],       # driving east
                     [[2,0,'L'], [2,1,'L'], FrenetPIDAgent]]  # oncoming driving west
else:
    raise NotImplemented("That scenario has not been made yet.")

COLORS        = ['186,0,0', '65,63,197'] # using colors from Audi.TT set.
assert(len(DYNAMIC_CARS) == len(COLORS))
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


    random.seed(0) # setting deterministic sampling of vehicle colors.

    for i,(start_goal_policy, color) in enumerate(zip(DYNAMIC_CARS, COLORS)):
        dyn_bp.set_attribute('color', color)
        start, goal, policy = start_goal_policy
        if i==0:
            start_pose = shift_pose_along_lane(INTERSECTION[start[0]][start[1]], -10.)
            goal_pose  = shift_pose_along_lane(INTERSECTION[goal[0]][goal[1]], 20.)
        else:
            start_pose = shift_pose_along_lane(INTERSECTION[start[0]][start[1]], -10.)
            goal_pose  = shift_pose_along_lane(INTERSECTION[goal[0]][goal[1]], 20.)

        if start[2] == 'L':
            start_pose = shift_pose_across_lane(start_pose)

        if goal[2]  == 'L':
            goal_pose = shift_pose_across_lane(goal_pose)

        start_transform = make_transform_from_pose(start_pose)
        goal_transform  = make_transform_from_pose(goal_pose)

        veh_actor   = world.spawn_actor(dyn_bp, start_transform)

        veh_policy  = policy(veh_actor,  goal_transform.location)

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
    bp_drone.set_attribute('fov', str(90))
    bp_drone.set_attribute('role_name', 'drone')
    drone = world.spawn_actor(bp_drone, cam_transform)

    return drone

def transform_to_local_frame(motion_hist_array):
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

def main():
    static_vehicle_list = []
    dynamic_vehicle_list = []
    dynamic_policy_list = []
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        world = client.reload_world() #client.get_world()
        if world.get_map().name != "Town05":
            world = client.load_world("Town05")
        world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

        static_vehicle_list = setup_static_cars(world)
        dynamic_vehicle_list, dynamic_policy_list = setup_dynamic_cars(world)
        drone = setup_camera(world)

        completed = False          # Flag to indicate when all cars have reached their destination
        fps = 20                   # FPS for the simulation under synchronous mode (TODO: finalize)
        use_spectator_view = False # Flag to indicate whether to overwrite default drone view with spectator view
        opencv_viz = True         # Flag to indicate whether to create an external window to view the drone view
        save_avi   = True        # Flag to indicate whether to save an avi of the drone view.

        # Predictions Setup
        agent_history = AgentHistory(world.get_actors())
        rasterizer    = SemBoxRasterizer(world.get_map().get_topology())
        pred_model    = DeployMultiPath(SAVEDMODELH5, ANCHORS)

        # Identify the target vehicle: the dynamic vehicle which is NOT ego.
        target_agent_id = []
        ego_agent_id    = []
        ego_policy      = []
        for vehicle, policy in zip(dynamic_vehicle_list, dynamic_policy_list):
            if type(policy) is not SMPCAgent:
                target_agent_id.append(vehicle.id)
            elif type(policy) is SMPCAgent:
                ego_agent_id.append(vehicle.id)
                ego_policy = policy
            else:
                raise ValueError(f"Invalid agent type: {policy}")
        assert len(target_agent_id) == 1, "Multiple target agents not supported at the moment!"
        target_agent_id = target_agent_id[0]
        assert len(ego_agent_id) == 1, "Multiple ego agents not supported at the moment!"
        ego_agent_id = ego_agent_id[0]

        with CarlaSyncMode(world, drone, fps=fps) as sync_mode:
            if use_spectator_view:
                spectator_transform = world.get_spectator().get_transform()
                drone.set_transform(spectator_transform)

            writer = None
            if save_avi:
                writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (960, 500))

            for veh_actor in dynamic_vehicle_list:
                # TODO: clean this up, quick hack:
                yaw_carla = veh_actor.get_transform().rotation.yaw
                carla_vel = carla.Vector3D(x=12.*np.cos(np.radians(yaw_carla)) ,
                                           y=12.*np.sin(np.radians(yaw_carla)) ,
                                           z=0.)
                veh_actor.set_target_velocity(carla_vel)

                snap, img = sync_mode.tick(timeout=2.0)

            debug_carla = world.debug
            colors = [carla.Color(r=255, g=0, b=255),
                      carla.Color(r=255, g=255, b=0),
                      carla.Color(r=0, g=255, b=255),
                      ]
            mean_box_extent = carla.Vector3D(x=0.2, y=0.2, z=0.2)
            box_rotation    = carla.Rotation(pitch=0., yaw=0., roll=0.)

            while not completed:
                snap, img = sync_mode.tick(timeout=2.0)
                agent_history.update(snap, world)

                # Handle drone view.
                img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
                img_array = np.reshape(img_array, (img.height, img.width, 4))
                img_array = img_array[:, :, :3]
                img_array = cv2.resize(img_array, (960, 500), interpolation = cv2.INTER_AREA)

                # Make predictions for the target vehicle (TODO: every time?)
                img_tv = rasterizer.rasterize(agent_history, target_agent_id, render_traffic_lights=False)
                past_states_tv, R_target_to_world, t_target_to_world = \
                    get_target_agent_history(agent_history, target_agent_id)

                curr_target_vehicle_position = R_target_to_world @ past_states_tv[-1, 1:3] + t_target_to_world
                target_vehicle_gmm_preds = []
                target_vehicle_positions = []
                if np.any(np.isnan(past_states_tv)):
                    # pass # Not enough data for predictions to be made.
                    target_vehicle_positions = [curr_target_vehicle_position]
                    target_vehicle_gmm_preds = [[np.stack([[curr_target_vehicle_position]*ego_policy.SMPC.N]*ego_policy.SMPC.N_modes)],
                                                [np.stack([[np.identity(2)]*ego_policy.SMPC.N]*ego_policy.SMPC.N_modes)]]
                    # import pdb; pdb.set_trace()
                    # print(len(target_vehicle_gmm_preds[0]))
                else:
                    gmm_pred_tv = pred_model.predict_instance(img_tv, past_states_tv[:-1])
                    gmm_pred_tv.transform(R_target_to_world, t_target_to_world)
                    gmm_pred_tv=gmm_pred_tv.get_top_k_GMM(ego_policy.SMPC.N_modes)
                    target_vehicle_gmm_preds = [[gmm_pred_tv.mus[:, :ego_policy.SMPC.N, :]], [gmm_pred_tv.sigmas[:, :ego_policy.SMPC.N, :, :]]]
                    target_vehicle_positions = [curr_target_vehicle_position]

                    for mean_traj, color in zip(target_vehicle_gmm_preds[0][0], colors):
                        for mean_pos in mean_traj:
                            loc = carla.Location(x=float(mean_pos[0]),
                                                 y=-float(mean_pos[1]),
                                                 z=float(0.2))
                            carla_box = carla.BoundingBox(loc, mean_box_extent)
                            debug_carla.draw_box(carla_box, box_rotation, color=color, life_time=ego_policy.dt)


                    # target_vehicle_positions = [curr_target_vehicle_position]
                    # target_vehicle_gmm_preds = [[np.stack([[curr_target_vehicle_position]*ego_policy.SMPC.N]*ego_policy.SMPC.N_modes)],
                                                # [np.stack([[np.identity(2)]*ego_policy.SMPC.N]*ego_policy.SMPC.N_modes)]]

                    # print(len(target_vehicle_gmm_preds[0]))
                    # import pdb; pdb.set_trace()
                # Handle updating the dynamic cars.  Terminate once all cars reach the goal.
                completed = True
                for act, policy in zip(dynamic_vehicle_list, dynamic_policy_list):
                    if type(policy) is SMPCAgent:
                        control = policy.run_step(target_vehicle_positions, target_vehicle_gmm_preds)

                        # For debugging:
                        vel   = act.get_velocity()
                        speed = ((vel.x**2) + (vel.y**2))**0.5
                        probs_str = f"EGO - v:{speed:.3f}, th: {control.throttle:.2f}, bk: {control.brake:.2f}, st: {control.steer:.2f}"
                        cv2.putText(img_array, probs_str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

                    else:
                        control = policy.run_step()
                    completed = completed and policy.done()
                    act.apply_control(control)

                # Handle OpenCV stuff.
                if opencv_viz:
                    cv2.imshow('Drone', img_array); cv2.waitKey(1)
                if save_avi:
                    writer.write(img_array)

            if save_avi:
                writer.release()

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
