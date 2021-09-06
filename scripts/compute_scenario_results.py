import os
import re
import glob
import pickle
import numpy as np
import pandas as pd

from evaluation.closed_loop_metrics import ScenarioResult, ClosedLoopTrajectory

def get_metric_dataframe(results_dir):
    scenario_dirs = sorted(glob.glob(results_dir + "*scenario*"))

    if len(scenario_dirs) == 0:
        raise ValueError(f"Could not detect scenario results in directory: {results_dir}")

    # Assumption: format is *scenario_<scene_num>_ego_init_<init_num>_policy
    dataframe = []
    for scenario_dir in scenario_dirs:
        scene_num = int( scenario_dir.split("scenario_")[-1].split("_")[0] )
        init_num  = int( scenario_dir.split("ego_init_")[-1].split("_")[0])
        policy    = re.split("ego_init_[0-9]*_", scenario_dir)[-1]

        pkl_path = os.path.join(scenario_dir, "scenario_result.pkl")
        if not os.path.exists(pkl_path):
            # TODO: should we be robust to directories without pkl files?
            raise RuntimeError(f"Unable to find a scenario_result.pkl in directory: {scenario_dir}")

        scenario_dict = pickle.load(open(pkl_path, "rb"))
        ego_entry   = [v for (k, v) in scenario_dict.items() if "ego" in k]
        tv_entries  = [v for (k, v) in scenario_dict.items() if "ego" not in k]

        assert len(ego_entry) == 1
        assert len(ego_entry) + len(tv_entries)  == len(scenario_dict.keys())

        sr = ScenarioResult( ego_closed_loop_trajectory = ClosedLoopTrajectory(**ego_entry[0]),
                             tv_closed_loop_trajectories = [ClosedLoopTrajectory(**v) for v in tv_entries])

        metrics_dict = sr.compute_metrics()
        dmins = metrics_dict.pop("dmins_per_TV")
        if dmins:
            metrics_dict["dmin_TV"] = np.amin(dmins) # take the closest distance to any TV in the scene
        else:
            metrics_dict["dmin_TV"] = np.nan
        metrics_dict["scenario"] = scene_num
        metrics_dict["initial"]  = init_num
        metrics_dict["policy"]   = policy
        dataframe.append(metrics_dict)

    return pd.DataFrame(dataframe)

def normalize_by_notv(df):
    scene_inits = set( [f"{s}_{i}" for (s,i) in zip(df.scenario, df.initial)])

    for scene_init in scene_inits:
        s, i = [int(x) for x in scene_init.split("_")]
        s_i_inds = np.logical_and(df.scenario == s, df.initial == i)
        notv_inds = np.logical_and(s_i_inds, df.policy=="notv")

        if np.sum(notv_inds) != 1:
            raise RuntimeError(f"Unable to find a unique notv execution for scenario {s}, initialization {i}.")

        notv_ind = np.where(notv_inds)[0].item()
        notv_lat_accel = df.max_lateral_acceleration[notv_ind]

        lat_accel_normalized = df[s_i_inds].max_lateral_acceleration / notv_lat_accel
        df.loc[s_i_inds, "max_lateral_acceleration"] = lat_accel_normalized

        # TODO: maybe we want to remove notv?
        # df.drop(notv_ind, inplace=True)

    df.rename(columns={"max_lateral_acceleration":
                       "max_lateral_acceleration_norm"},
              inplace=True)

    return df

def aggregate(df):
    # TODO: check we've run all the scenarios we need to first.

    df_aggregate = []

    for scenario in set(df.scenario):
        for policy in set(df.policy):
            subset_inds = np.logical_and( df.scenario == scenario, df.policy == policy )
            # TODO: how many subset_inds do we expect there to be?

            res = df[subset_inds].mean(numeric_only=True)
            res.drop(["initial", "scenario"], inplace=True)

            res_dict = {"scenario": int(scenario), "policy": policy}
            res_dict.update(res.to_dict())
            df_aggregate.append(res_dict)

    return pd.DataFrame(df_aggregate)



if __name__ == '__main__':
    results_dir = os.path.join(os.path.abspath(__file__).split('scripts')[0], 'results/')

    dataframe = get_metric_dataframe(results_dir)
    dataframe.to_csv(os.path.join(results_dir, "df_full.csv"), sep=",")

    dataframe = normalize_by_notv(dataframe)
    dataframe.to_csv(os.path.join(results_dir, "df_norm.csv"), sep=",")

    dataframe  = aggregate(dataframe)
    dataframe.to_csv(os.path.join(results_dir, "df_final.csv"), sep=",")
