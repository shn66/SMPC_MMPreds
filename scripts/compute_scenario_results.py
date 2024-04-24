import os
import re
import glob
import numpy as np
import pandas as pd
import pdb
import matplotlib
import pickle as pkl
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from evaluation.closed_loop_metrics import ScenarioResult, ClosedLoopTrajectory, load_scenario_result

def get_metric_dataframe(results_dir):
    scenario_dirs = sorted(glob.glob(results_dir + "*scenario_lk*"))

    if len(scenario_dirs) == 0:
        raise ValueError(f"Could not detect scenario results in directory: {results_dir}")

    # Assumption: format is *scenario_<scene_num>_ego_init_<init_num>_policy
    dataframe = []
    for scenario_dir in scenario_dirs:
        # pdb.set_trace()
        scene_num =  1#scenario_dir.split("scenario_")[-1].split("_")[0] 
        init_num  = int( scenario_dir.split("ego_init_")[-1].split("_")[0])
        policy    = re.split("ego_init_[0-9]*_", scenario_dir)[-1]

        pkl_path = os.path.join(scenario_dir, "scenario_result.pkl")

        
        if not os.path.exists(pkl_path):
            raise RuntimeError(f"Unable to find a scenario_result.pkl in directory: {scenario_dir}")

        notv_pkl_path = os.path.join(re.split(f"{policy}", scenario_dir)[0] + "notv", "scenario_result.pkl")
        if not os.path.exists(notv_pkl_path):
            raise RuntimeError(f"Unable to find a notv scenario_result.pkl in location: {notv_pkl_path}")

        # Load scenario dict for this policy and the notv case (for Hausdorff distance).
        sr      = load_scenario_result(pkl_path)
        notv_sr = load_scenario_result(notv_pkl_path)

        metrics_dict = sr.compute_metrics()
        metrics_dict["hausdorff_dist_notv"] = sr.compute_ego_hausdorff_dist(notv_sr)
        dmins = metrics_dict.pop("dmins_per_TV")
        if dmins:
            metrics_dict["dmin_TV"] = np.amin(dmins) # take the closest distance to any TV in the scene
        else:
            metrics_dict["dmin_TV"] = np.nan # no moving TVs in the scene
        metrics_dict["scenario"] = scene_num
        metrics_dict["initial"]  = init_num
        metrics_dict["policy"]   = policy
        dataframe.append(metrics_dict)

    return pd.DataFrame(dataframe)

def make_trajectory_viz_plot(results_dir, color1="r", color2="b", plot_init=1, plot_pol="no_switch"):
    scenario_dirs = sorted(glob.glob(results_dir + "*scenario*lk*"))

    if len(scenario_dirs) == 0:
        raise ValueError(f"Could not detect scenario results in directory: {results_dir}")

    # Assumption: format is *scenario_<scene_num>_ego_init_<init_num>_policy
    dataframe = []
    for scenario_dir in scenario_dirs:
        scene_num = 1#scenario_dir.split("scenario_")[-1].split("_")[0]
        init_num  = int( scenario_dir.split("ego_init_")[-1].split("_")[0])
        policy    = re.split("ego_init_[0-9]*_", scenario_dir)[-1]

        pkl_path = os.path.join(scenario_dir, "scenario_result.pkl")

        if not os.path.exists(pkl_path):
            raise RuntimeError(f"Unable to find a scenario_result.pkl in directory: {scenario_dir}")

        notv_pkl_path = os.path.join(re.split(f"{policy}", scenario_dir)[0] + "notv", "scenario_result.pkl")
        if not os.path.exists(notv_pkl_path):
            raise RuntimeError(f"Unable to find a notv scenario_result.pkl in location: {notv_pkl_path}")
        
        notv_cl_pkl_path = os.path.join(re.split(f"{policy}", scenario_dir)[0] + "notv_cl", "scenario_result.pkl")
        if not os.path.exists(notv_pkl_path):
            raise RuntimeError(f"Unable to find a notv_cl scenario_result.pkl in location: {notv_pkl_path}")

        # Load scenario dict for this policy and the notv case (for Hausdorff distance).
        if init_num==plot_init:
            if"no_switch" in scenario_dir:
                sr      = load_scenario_result(pkl_path)
                notv_sr = load_scenario_result(notv_pkl_path)
                notv_cl_sr = load_scenario_result(notv_cl_pkl_path)

                # Get time vs. frenet projection for this policy's ego trajectory vs the notv case.
                ts, s_wrt_notv, ey_wrt_notv, epsi_wrt_notv = sr.compute_ego_frenet_projection(notv_sr)

                # Get time vs. frenet projection for this policy's ego trajectory vs cl.
                ts_cl, s_wrt_cl, ey_wrt_cl, epsi_wrt_cl = sr.compute_ego_frenet_projection(notv_cl_sr)
                v=sr.ego_closed_loop_trajectory.state_trajectory[:,-1]
                a=sr.ego_closed_loop_trajectory.input_trajectory[:,0]
                steer=sr.ego_closed_loop_trajectory.input_trajectory[:,-1]

            elif "open_loop" in scenario_dir:

                sr_ol      = load_scenario_result(pkl_path)
                notv_sr = load_scenario_result(notv_pkl_path)
                notv_cl_sr = load_scenario_result(notv_cl_pkl_path)



                # Get time vs. frenet projection for this policy's ego trajectory vs cl.
                ts_cl, s_ol_wrt_cl, ey_ol_wrt_cl, epsi_ol_wrt_cl = sr_ol.compute_ego_frenet_projection(notv_cl_sr)
                v_ol=sr_ol.ego_closed_loop_trajectory.state_trajectory[:,-1]
                a_ol=sr_ol.ego_closed_loop_trajectory.input_trajectory[:,0]
                steer_ol=sr_ol.ego_closed_loop_trajectory.input_trajectory[:,-1]
        

            
            # # Get the closest distance to a TV across all timesteps identified above.
            # d_closest = np.ones(ts.shape) * np.inf
            # d_trajs_TV = sr.get_distances_to_TV()

            # for tv_ind in range(len(d_trajs_TV)):
            #     t_traj = d_trajs_TV[tv_ind][:,0]
            #     d_traj = d_trajs_TV[tv_ind][:,1]

            #     d_interp = np.interp(ts, t_traj, d_traj, left=np.inf, right=np.inf)

            #     d_closest = np.minimum(d_interp, d_closest)

            # Make the plots.
            # t0 = sr.ego_closed_loop_trajectory.state_trajectory[0, 0]
            # trel = ts - t0
            # ax1 = plt.gca()
            # ax3.plot(np.array(s_cl)-s_cl[0], v_cl, 'b', linewidth=2.0, label="Ours")
            # ax3.plot(np.array(s_ol)-s_ol[0], v_ol, 'r', linewidth=2.0, label="BL")
            # ax3.set_ylabel("Speed")
            # ax3.set_xlabel("$s$")
            # plt.legend()
            # ax1.set_xlabel("Time (s)")
            # ax1.set_ylabel("Route Progress (m)", color=color1)
            # ax1.plot(trel[::2], s_wrt_notv[::2], color=color1)
            # ax1.tick_params(axis="y", labelcolor=color1)
            # ax1.set_yticks(np.arange(0., 101., 10.))

            # ax2 = ax1.twinx()
            # ax2.set_ylabel("Closest TV distance (m)", color=color2)
            # ax2.plot(trel[::2], d_closest[::2], color=color2)
            # ax2.tick_params(axis="y", labelcolor=color2)
            # ax2.set_yticks(np.arange(0., 51., 5.))

            # ax1.plot(s_wrt_cl, ey_wrt_cl)

            # plt.tight_layout()
            # plt.savefig(f'{scenario_dir}/traj_viz.svg', bbox_inches='tight')
    fig=plt.figure(figsize=(10,15))
    
    
    data_cl={'s':np.array(s_wrt_cl)[:-64]-s_wrt_cl[0],
             'ey_ref':np.ones(len(s_wrt_cl)-64)*3.6,
             'ey':np.array(ey_wrt_cl)[:-64].squeeze(),
             'epsi':180/np.pi*np.array(epsi_wrt_cl)[:-64].squeeze(),
             'v':np.convolve(np.array(v).squeeze(),np.ones(20)*0.05,mode='same')[:-64],
             'a':np.convolve(np.array(a).squeeze(),np.ones(20)*0.05,mode='same')[:-64],
             'steer':np.convolve(180/np.pi*np.array(steer).squeeze(),np.ones(5)*0.2,mode='same')[:-64]}
    
    data_ol={'s':np.array(s_ol_wrt_cl)[:-2]-s_ol_wrt_cl[0],
             'ey_ref':np.ones(len(s_ol_wrt_cl)-2)*3.6,
             'ey':np.array(ey_ol_wrt_cl)[:-2].squeeze(),
             'epsi':180/np.pi*np.array(epsi_ol_wrt_cl)[:-2].squeeze(),
             'v':np.convolve(np.array(v_ol).squeeze(),np.ones(20)*0.05,mode='same')[:-2],
             'a':np.convolve(np.array(a_ol).squeeze(),np.ones(20)*0.05,mode='same')[:-2],
             'steer':np.convolve(180/np.pi*np.array(steer_ol).squeeze(),np.ones(5)*0.2,mode='same')[:-2]}
    
    # dicts=[data_cl, data_ol]
    # with open('cl_vs_ol_data.pkl', 'wb') as f: 
    #     pkl.dump(dicts, f, protocol=pkl.HIGHEST_PROTOCOL)
    
   

    ax1=plt.subplot(511)
    ax1.plot(np.array(s_wrt_cl)[:-64]-s_wrt_cl[0], np.ones(len(s_wrt_cl)-64)*3.6, 'k--', linewidth=1.5, label="$e_y^{ref}$")
    ax1.plot(np.array(s_wrt_cl)[:-64]-s_wrt_cl[0], np.convolve(np.array(ey_wrt_cl).squeeze(),np.ones(5)*0.2,mode='same')[:-64], 'b', linewidth=2.0, label="Proposed")
    ax1.plot(np.array(s_ol_wrt_cl)[:-2]-s_ol_wrt_cl[0], np.convolve(np.array(ey_ol_wrt_cl).squeeze(),np.ones(5)*0.2,mode='same')[:-2], 'r--', linewidth=2.0, label="OL")
    # ax1.plot(np.array(s_ol)-s_ol[0], np.array(ey_ol).squeeze(), 'r', linewidth=2.0, label="BL")
    ax1.set_ylabel("$e_y [m]$")
    plt.grid()
    plt.legend()
    ax2=plt.subplot(512)
    # ax1.plot(np.array(s_wrt_cl)-s_wrt_cl[0], np.ones(len(s_wrt_cl))*3.5, 'k--', linewidth=1.5, label="$e_y^{ref}$")
    ax2.plot(np.array(s_wrt_cl)[:-64]-s_wrt_cl[0], 180/np.pi*np.array(epsi_wrt_cl)[:-64].squeeze(), 'b', linewidth=2.0, label="Proposed")
    ax2.plot(np.array(s_ol_wrt_cl)[:-2]-s_ol_wrt_cl[0], 180/np.pi*np.array(epsi_ol_wrt_cl).squeeze()[:-2], 'r--', linewidth=2.0, label="OL")
    ax2.set_ylabel("$e_\psi [deg]$ ")
    plt.grid()
    # plt.legend()
    ax3=plt.subplot(513)
    ax3.plot(np.array(s_wrt_cl)[:-64]-s_wrt_cl[0], np.convolve(np.array(v).squeeze(),np.ones(20)*0.05,mode='same')[:-64], 'b', linewidth=2.0, label="Proposed")
    ax3.plot(np.array(s_ol_wrt_cl)[:-2]-s_ol_wrt_cl[0], np.convolve(np.array(v_ol).squeeze(),np.ones(20)*0.05,mode='same')[:-2], 'r--', linewidth=2.0, label="OL")
    ax3.set_ylabel("Speed $[m/s]$")
    plt.grid()
    ax4=plt.subplot(514)
    ax4.plot(np.array(s_wrt_cl)[:-64]-s_wrt_cl[0], np.convolve(180/np.pi*np.array(steer).squeeze(),np.ones(2)*0.5,mode='same')[:-64], 'b', linewidth=2.0, label="Proposed")
    ax4.plot(np.array(s_ol_wrt_cl)[:-2]-s_ol_wrt_cl[0], np.convolve(180/np.pi*np.array(steer_ol).squeeze(),np.ones(2)*0.5,mode='same')[:-2], 'r--', linewidth=2.0, label="OL")
    ax4.set_ylabel("Steering [deg]")
    plt.grid()
    ax5=plt.subplot(515)
    ax5.plot(np.array(s_wrt_cl)[:-64]-s_wrt_cl[0], np.convolve(np.array(a).squeeze(),np.ones(20)*0.05,mode='same')[:-64], 'b', linewidth=2.0, label="Proposed")
    ax5.plot(np.array(s_ol_wrt_cl)[:-2]-s_ol_wrt_cl[0], np.convolve(np.array(a_ol).squeeze(),np.ones(20)*0.05,mode='same')[:-2], 'r--', linewidth=2.0, label="OL")
    ax5.set_ylabel("$a [m/s^2]$")
    ax5.set_xlabel("$Station[m]$")
    plt.grid()
    

    plt.show()

    # fig.savefig('traj_viz.png', bbox_inches='tight')
            
def normalize_by_notv(df):
    # Compute metrics that involve normalizing by the notv scenario execution.
    # Right now, these metrics are completion_time and max_lateral_acceleration.

    # Add the new columns with normalized values.
    df = df.assign( max_lateral_acceleration_norm = df.max_lateral_acceleration,
                    completion_time_norm = df.completion_time)

    # Do the normalization per scenario / ego initial condition.
    scene_inits = set( [f"{s}_{i}" for (s,i) in zip(df.scenario, df.initial)])

    for scene_init in scene_inits:
        s, i = [int(float(x)) for x in scene_init.split("_")]
        s_i_inds = np.logical_and(df.scenario == s, df.initial == i)
        notv_inds = np.logical_and(s_i_inds, df.policy=="notv")

        if np.sum(notv_inds) != 1:
            raise RuntimeError(f"Unable to find a unique notv execution for scenario {s}, initialization {i}.")

        notv_ind       = np.where(notv_inds)[0].item()
        notv_lat_accel = df.max_lateral_acceleration[notv_ind]
        notv_time      = df.completion_time[notv_ind]

        lat_accel_normalized = df[s_i_inds].max_lateral_acceleration / notv_lat_accel
        df.loc[s_i_inds, "max_lateral_acceleration_norm"] = lat_accel_normalized

        time_normalized = df[s_i_inds].completion_time / notv_time
        df.loc[s_i_inds, "completion_time_norm"] = time_normalized

    return df

def aggregate(df):
    df_aggregate = []

    for scenario in set(df.scenario):
        for policy in set(df.policy):
            subset_inds = np.logical_and( df.scenario == scenario, df.policy == policy )

            res = df[subset_inds].mean(numeric_only=True)
            res.drop(["initial", "scenario"], inplace=True)

            res_dict = {"scenario": int(scenario), "policy": policy}
            res_dict.update(res.to_dict())
            df_aggregate.append(res_dict)

    return pd.DataFrame(df_aggregate)

if __name__ == '__main__':
    compute_metrics = False
    make_traj_viz   = True
    results_dir = os.path.join(os.path.abspath(__file__).split('scripts')[0], 'results/')

    if compute_metrics:
        dataframe = get_metric_dataframe(results_dir)
        dataframe.to_csv(os.path.join(results_dir, "df_full.csv"), sep=",")

        dataframe = normalize_by_notv(dataframe)
        dataframe.to_csv(os.path.join(results_dir, "df_norm.csv"), sep=",")

        dataframe  = aggregate(dataframe)
        dataframe.to_csv(os.path.join(results_dir, "df_final.csv"), sep=",")

    if make_traj_viz:
        make_trajectory_viz_plot(results_dir)
