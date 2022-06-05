

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pdb
from random import sample


class Simulator():

    def __init__(self,
#                 EV_init     = np.array([0., 12.]),
#                 TV_init     = [np.array([-15., 16.])],
#                 G_TV_init   = [2.],
                EV_init     = np.array([-40., 0., 10]),
                TV_init     = [np.array([25., 35., -15.]), np.array([36., -40., 13.])],
                DT          = 0.1,
                T_FINAL     = 500,
                S_FINAL     = 50,
                O_FINAL     = 20,
                O_STOP      = 6.,
                NOISE_STD   =  [0.001, 0.001, 0.2, 0.25]
                ):

        self.ev_init=EV_init
        self.N_TV=len(TV_init)
        self.tv_init= [TV_init[k] for k in range(self.N_TV)]
        self.gains=[[.1,0.2], [3,2]]
        self.N_modes=4

        self.t=0
        self.dt= DT
        self.T=T_FINAL
        self.s_f=S_FINAL
        self.o_f=O_FINAL
        self.o_s=O_STOP
        self.mode=0

        self.A=np.array([[1., 0., self.dt],[0., 1., 0.], [0., 0., 1.]])
        self.B=np.array([0.,0.,self.dt])

        self.Atv=np.array([[1., 0., 0.],[0., 1., self.dt], [0., 0., 1.]])
        self.Btv=self.B

        self.ev_traj=np.zeros((3,self.T+1))
        self.ev_u=np.zeros((1,self.T))
        self.ev_traj[:,0]=self.ev_init

        self.tv_traj=[np.zeros((3,self.T+1)) for k in range(self.N_TV)]

        for k in range(self.N_TV):
            self.tv_traj[k][:,0]=self.tv_init[k]

        self.u_prev=0.
        self.noise_std=NOISE_STD



    def TV_gen(self):
            if self.tv_traj[0][1,self.t]< -self.o_f:
#                 self.tv_traj[0]=np.zeros((3,self.T+1))
                self.tv_traj[0][:,self.t]=self.tv_init[0]
                mode_list=[(self.mode+2)%4, self.mode]
#                 self.mode=sample(mode_list,1)[0]
            if self.tv_traj[1][1,self.t]> self.o_f:
#                 self.tv_traj[1]=np.zeros((3,self.T+1))
                self.tv_traj[1][:,self.t]=self.tv_init[1]
                mode_list=[self.mode+(-1)**(self.mode), self.mode]
#                 self.mode=sample(mode_list,1)[0]
                self.mode=self.mode+(-1)**(self.mode)

    def done(self):
        return self.t==self.T or self.s_f-self.ev_traj[0,self.t]<=0.1

    def get_update_dict(self, N, *args):

        if self.t==0:
            x_lin=[np.zeros((3,N+1))]*self.N_modes
            u_tvs=[[np.zeros(N) for j in range(self.N_modes)] for k in range(self.N_TV)]
        else:
            x_lin, u_tvs =self._get_lin_ref(N, *args)


        update_dict={'x0': self.ev_traj[0,self.t], 'y0': self.ev_traj[1,self.t], 'v0':self.ev_traj[2,self.t], 'u_prev':self.u_prev,
                     'x_tv0': np.array([self.tv_traj[k][0,self.t] for k in range(self.N_TV)]), 'y_tv0': np.array([self.tv_traj[k][1,self.t] for k in range(self.N_TV)]), 'v_tv0': np.array([self.tv_traj[k][2,self.t] for k in range(self.N_TV)]),
                     'x_lin': x_lin, 'u_tvs': u_tvs}
        if len(args)!=0:
            if len(args)==4:
                [h,M,K,nom_z_tv]=args
                update_dict.update({'ws':[h,M,K]})
            else:
                [h,M,K,lmbd,nu,nom_z_tv]=args
                update_dict.update({'ws':[h,M,K,lmbd,nu]})

        return update_dict


    def run_step(self, u_ev):
#         pdb.set_trace()
        rng=np.random.default_rng(self.t)
#         if self.ev_traj[0,self.t]>=self.s_f-0.35 and self.mode==1:
#             u_ev=-0.1
        self.ev_traj[:,self.t+1]=self.A@self.ev_traj[:,self.t]+self.B*u_ev\
                                    +np.array([rng.normal(0,self.noise_std[0]), 0., rng.normal(0,self.noise_std[1])])
        self.ev_traj[2,self.t+1]=np.max([.0, self.ev_traj[2,self.t+1] ])
        self.ev_u[:,self.t]=u_ev
        self.u_prev=u_ev
        u_tv=[]
        if int(self.mode/2)==0:
            u_tv.append(self.gains[0][0]*( -self.o_f-self.tv_traj[0][1,self.t])+self.gains[0][1]*( 0.-self.tv_traj[0][2,self.t]))
        else:
            u_tv.append(self.gains[1][0]*( self.o_s-self.tv_traj[0][1,self.t])+self.gains[1][1]*( 0.-self.tv_traj[0][2,self.t]))
        if self.mode%2==0:
            u_tv.append(self.gains[0][0]*( self.o_f-self.tv_traj[0][1,self.t])+self.gains[0][1]*( 0.-self.tv_traj[0][2,self.t]))
        else:
            u_tv.append(self.gains[1][0]*( -self.o_s-self.tv_traj[1][1,self.t])+self.gains[1][1]*( 0.-self.tv_traj[1][2,self.t]))

        for k in range(self.N_TV):

            self.tv_traj[k][:,self.t+1]=self.Atv@self.tv_traj[k][:,self.t]+self.Btv*u_tv[k]\
                                            +np.array([0., rng.normal(0,self.noise_std[2]), rng.normal(0,self.noise_std[3])])

#         pdb.set_trace()
        self.t+=1
        self.TV_gen()


    def _get_lin_ref(self, N, *args):#h_opt, M_opt, K_opt, nom_z_tv):
#         R=self._rot2(self.semiaxes_theta)


#         pdb.set_trace()
        w=np.diag(np.array(self.noise_std[0:2])**(-1))@(self.ev_traj[0::2,self.t]-self.A[0::2,:]@self.ev_traj[:,self.t-1]-self.B[0::2]*self.u_prev)


        x0=[self.ev_traj[:,self.t] for j in range(self.N_modes)]
        o0=[[self.tv_traj[k][:,self.t] for k in range(self.N_TV)] for j in range(self.N_modes)]
        elim_stop=False
        if o0[0][0][1]>0:
            elim_stop=True
        x_lin=[np.zeros((3,N+1)) for j in range(self.N_modes)]
        u_tvs=[[np.zeros(N) for j in range(self.N_modes)] for k in range(self.N_TV)]
        for j in range(self.N_modes):
            x=x0[j]
            o=o0[j]
            x_lin[j][:,0]=x
            w_seq=np.zeros(2*N)
            w_seq[:2]=w

            if len(args)==0:
                h_opt=[np.zeros((N,1))]*self.N_modes
                M_opt=[np.zeros((N,2*N))]*self.N_modes
                K_opt=[[np.zeros((N,2*N))]*self.N_TV]*self.N_modes
            else:
                if len(args)==4:
                    [h_opt,M_opt,K_opt,nom_z_tv]=args
                else:
                    [h_opt,M_opt,K_opt,lmbd,nu,nom_z_tv]=args

                for i in range(1,N):

                    u=h_opt[j][i]+M_opt[j][i,:]@w_seq+np.sum([K_opt[j][l][i,2*i:2*(i+1)]@(o[l][1:]-nom_z_tv[j][l][3*i+1:3*(i+1)]) for l in range(self.N_TV)])
                    if i==1:
                        self.u_prev=u
                    if int(j/2)==0 or elim_stop:
                        u_tvs[0][j][i-1]=self.gains[0][0]*( -self.o_f-o[0][1])+self.gains[0][1]*( -0-o[0][2])
                    else:
                        u_tvs[0][j][i-1]=self.gains[1][0]*( self.o_s-o[0][1])+self.gains[1][1]*( -0-o[0][2])
                    if j%2==0:
                        u_tvs[1][j][i-1]=self.gains[0][0]*( self.o_f-o[1][1])+self.gains[0][1]*( -0-o[1][2])
                    else:
                        u_tvs[1][j][i-1]=self.gains[1][0]*( -self.o_s-o[1][1])+self.gains[1][1]*( 0-o[1][2])

                    o=[self.Atv@o[k]+self.Btv*u_tvs[k][j][i-1] for k in range(self.N_TV)]
                    if i==N-1:
                        if int(j/2)==0:
                            u_tvs[0][j][i]=self.gains[0][0]*( -self.o_f-o[0][1])+self.gains[0][1]*( 0-o[0][2])
                        else:
                            u_tvs[0][j][i]=self.gains[1][0]*( self.o_s-o[0][1])+self.gains[1][1]*( 0-o[0][2])
                        if j%2==0:
                            u_tvs[1][j][i]=self.gains[0][0]*( self.o_f-o[1][1])+self.gains[0][1]*( 0-o[1][2])
                        else:
                            u_tvs[1][j][i]=self.gains[1][0]*( -self.o_s-o[1][1])+self.gains[1][1]*( 0-o[1][2])

                    x=self.A@x+self.B*u
                    x_lin[j][:,i]=x
                    x_lin[j][:,i+1]=x
#                 print(x_lin[j])
#                 print(nom_z_tv[j])
        return x_lin, u_tvs



