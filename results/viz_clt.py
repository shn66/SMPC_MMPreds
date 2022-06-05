import pickle
import matplotlib.pyplot as plt

scene_dir = "scenario_03_ego_init_01_smpc_no_switch"

d = pickle.load(open(f"{scene_dir}/scenario_result.pkl", "rb"))
ego_key = [k for k in d.keys() if "ego" in k][0]

ev = d[ego_key]
st = ev["state_trajectory"]
it = ev["input_trajectory"]

# State Trajectory
plt.subplot(411)
plt.plot(st[:,0], st[:,1])
plt.subplot(412)
plt.plot(st[:,0], st[:,2])
plt.subplot(413)
plt.plot(st[:,0], st[:,3])
plt.subplot(414)
plt.plot(st[:,0], st[:,4])

# Input Trajectory
plt.figure()
plt.subplot(211)
plt.plot(st[:,0], it[:,0])
plt.subplot(212)
plt.plot(st[:,0], it[:,1])

plt.show()
