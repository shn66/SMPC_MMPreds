import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pdb
from random import sample
from sim import Simulator
from SMPC_OBCA import SMPC_MMPreds_OBCA




if __name__=="__main__":
	Sim=Simulator()
	smpc = SMPC_MMPreds_OBCA(DT=Sim.dt, NOISE_STD=Sim.noise_std)
	# smpc = SMPC_MMPreds_ACC_IA(DT=Sim.dt, NOISE_STD=Sim.noise_std, Estimate=True)
	is_opt=False
	ctr=0
	feas=0
	opt_list=[]
	i=7
	while not Sim.done():
	    if Sim.t==0 or not is_opt:
	        update_dict=Sim.get_update_dict(smpc.N)
	    else:
	        update_dict=Sim.get_update_dict(smpc.N, sol_dict['h_opt'], sol_dict['M_opt'], sol_dict['K_opt'], sol_dict['nom_z_tv'])

	    smpc.update(i,update_dict)
	    sol_dict=smpc.solve(i)
	    is_opt=sol_dict['optimal']
	    Sim.run_step(sol_dict['u_control'])
	#     Sim.run_step(0.)
	    print([sol_dict['solve_time'], is_opt, Sim.ev_traj[:,Sim.t-1], Sim.tv_traj[0][:,Sim.t-1], Sim.tv_traj[1][:,Sim.t-1]])
	    feas+=int(is_opt)
	    opt_list.append(is_opt)
	    ctr+=1

