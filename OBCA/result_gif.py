import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from celluloid import Camera
from sim import Simulator
import pdb
import os
import glob

[Sim,Sim_ol] = pkl.load( open( os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "save.p" ), "rb" ) )
# pdb.set_trace()
fig= plt.figure()
camera = Camera(fig)

for i in range(Sim_ol.t+1):
    plt.plot([-50, 20], [5, 5], color='k', lw=2)
    plt.plot([-50, 20], [-5, -5], color='k', lw=2)
    plt.plot([40, 50], [5, 5], color='k', lw=2)
    plt.plot([40, 50], [-5, -5], color='k', lw=2)
    plt.plot([20, 20], [5, 30], color='k', lw=2)
    plt.plot([40, 40], [5, 30], color='k', lw=2)
    plt.plot([20, 20], [-5, -30], color='k', lw=2)
    plt.plot([40, 40], [-5, -30], color='k', lw=2)
    lin_obca,=plt.plot(Sim.ev_traj[0,:i], Sim.ev_traj[1,:i], color='green', lw=3)
    lin_ellipse,=plt.plot(Sim_ol.ev_traj[0,:i], Sim_ol.ev_traj[1,:i], color='blue', lw=3, alpha=0.5)
#     plt.plot(Sim.tv_traj[0][0,:i], Sim.tv_traj[0][1,:i], color='orange')
#     plt.plot(Sim.tv_traj[1][0,:i], Sim.tv_traj[1][1,:i], color='orange')
    plt.gca().add_patch(Rectangle((Sim_ol.tv_traj[0][0,i-1]-1.8,Sim_ol.tv_traj[0][1,i-1]-2.9),3.6,5.8,linewidth=1,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((Sim_ol.tv_traj[1][0,i-1]-1.8,Sim_ol.tv_traj[1][1,i-1]-2.9),3.6,5.8,linewidth=1,edgecolor='r',facecolor='none'))
    if i<=Sim.t:
        plt.gca().add_patch(Rectangle((Sim.ev_traj[0,i-1]-2.9,Sim.ev_traj[1,i-1]-1.8),5.8,3.6,linewidth=1,edgecolor='g',facecolor='none'))
    plt.gca().add_patch(Rectangle((Sim_ol.ev_traj[0,i-1]-2.9,Sim_ol.ev_traj[1,i-1]-1.8),5.8,3.6,linewidth=1,edgecolor='b',facecolor='none', alpha=0.5))
    plt.axis('equal')
    plt.title("SMPC: Linearized OBCA vs Linearised Ellipse")
    plt.legend([lin_obca, lin_ellipse], ["OBCA EV speed=%s "%(Sim.ev_traj[2,i-1]),"ellipse EV speed=%s"%(Sim_ol.ev_traj[2,i-1])])
    camera.snap()

animation = camera.animate()
animation.save(os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "anim.mp4" ))