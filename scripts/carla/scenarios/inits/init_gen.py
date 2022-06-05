import numpy as np
import json



rng=np.random.default_rng(123)
vel_inits=9.0+(rng.random(50)-0.5)*2
long_inits=-15.0+(rng.random(50)-0.5)*5


init_dict={"start_longitudinal_offset" : 0.0, "init_speed" : 0.0}


for i in range(50):
	if i<=8:
		json_name=f"ego_init_0{i+1}.json"
	else:
		json_name=f"ego_init_{i+1}.json"
	init_dict["start_longitudinal_offset"]=long_inits[i]
	init_dict["init_speed"]=vel_inits[i]
	with open(json_name, "w") as outfile:
		json.dump(init_dict, outfile)





