import time
import casadi as ca
import numpy as np
import pdb

class SMPC_MMPreds_OBCA():

#     def __init__(self,
#                 N            =  20,
#                 DT           = 0.1,
#                 V_MIN        = 0.0,
#                 V_MAX        = 15.0,
#                 A_MIN        = -7.0,
#                 A_MAX        =  3.0,
#                 N_modes_MAX  =  1,
#                 N_TV         =  1,
#                 T_BAR_MAX    =  7,
#                 D_NOM        =  7,
#                 S_FINAL      =  60,
#                 TIGHTENING   =  1.6,
#                 NOISE_STD    =  [0.01, 0.01, 0.02, .05, .5], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
#                 Q = [100., 100.], # weights on x, y, and v.
#                 R = 100.,       # weights on inputs
#                 Estimate=False
#                 ):
    def __init__(self,
                N            =  12,
                DT           = 0.1,
                V_MIN        = -0.5,
                V_MAX        = 12.0,
                A_MIN        = -8.0,
                A_MAX        =  5.0,
                N_modes_MAX  =  4,
                N_TV         =  2,
                T_BAR_MAX    =  8,
                D_NOM        =  1.,
                S_FINAL      =  100,
                O_FINAL      =  20,
                O_STOP       =  6.,
                TIGHTENING   =  1.8,
                NOISE_STD    =  [0.001, 0.001, 0.002, 0.002], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                Q = 200., # weights on x, y, and v.
                R = 100.,       # weights on inputs
                OL_FLAG     = False
                ):
        self.N=N
        self.DT=DT
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.A_MAX=A_MAX
        self.A_MIN=A_MIN
        self.N_modes=N_modes_MAX
        self.N_TV=N_TV
        self.t_bar_max=T_BAR_MAX
#         self.t_bar_max=self.N-1
        self.d_nom=D_NOM
        self.s_f=S_FINAL
        self.o_f=O_FINAL
        self.o_s=O_STOP
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.OL=OL_FLAG
        self.A=ca.DM([[1., 0., self.DT],[0., 1., 0.], [0., 0., 1.]])
        self.B=ca.DM([0.,0.,self.DT])

        self.Rtv=ca.DM([[0.,1.],[-1.,0.]])
        self.G=ca.DM([[1., 0.],[-1.,0. ], [0, 1.],[0.,-1. ]])
        self.g=ca.DM([[3.2],[3.2], [1.8],[1.8]])

        self.Atv=ca.DM([[1., 0., 0.],[0., 1., self.DT], [0., 0., 1.]])
        self.Btv=self.B

        self.gains=[ca.DM([[0,.1,0.2]]), ca.DM([[0,3,2]])]


        self.nom_z_tv=[]
        self.oa_constr=[]
        self.oa_lconstr=[]

        self.opti=[]

        self.u_backup=[]
        self.z_curr=[]
        self.u_prev=[]
        self.h_prev=[]
        self.M_prev=[]
        self.K_prev=[]
        self.lmbd_prev=[]
        self.nu_prev=[]
        self.mode_list=[]

        self.lmbd_dual_var=[]
        self.nu_dual_var=[]

        self.z_lin=[]
        self.u_tvs=[]
        self.z_tv_curr=[]
#         self.term_flag=[]
        self.policy=[]

#         p_opts = {'expand': True}
#         s_opts = {'do_super_scs': 0, 'eps':1e-3, 'max_iters':300}
#         s_opts_grb = {'OutputFlag': 1, 'FeasibilityTol' : 1e-2, 'PSDTol' : 1e-3, 'BarConvTol':1e-5}
        s_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-2, 'PSDTol' : 1e-3}
#         s_opts_grb = {'OutputFlag': 0}
        p_opts_grb = {'error_on_fail':0}
#         p_opts = {'expand': True}
#         s_opts = {'print_level': 1}

        for i in range(self.t_bar_max+1):
#         for i in range(3,4):
            self.opti.append(ca.Opti('conic'))
#             self.opti.append(ca.Opti())
#             self.opti[i].solver('ipopt', p_opts, s_opts)
            self.opti[i].solver("gurobi", p_opts_grb, s_opts_grb)

#             self.opti[i].solver("superscs", p_opts_grb,s_opts)

            self.z_curr.append(self.opti[i].parameter(3))
            self.u_prev.append(self.opti[i].parameter(1))
#             self.term_flag.append(self.opti[i].parameter(1))

            N_TV=self.N_TV
            t_bar=i


            self.z_lin.append([self.opti[i].parameter(3,self.N+1) for j in range(self.N_modes)])
            self.u_tvs.append([[self.opti[i].parameter(self.N,1) for j in range(self.N_modes)] for k in range(N_TV)])
            self.z_tv_curr.append(self.opti[i].parameter(3*N_TV))
            self.h_prev.append([self.opti[i].parameter(self.N,1) for j in range(self.N_modes)])
            self.M_prev.append([self.opti[i].parameter(self.N,2*self.N) for j in range(self.N_modes)])
            self.K_prev.append([[self.opti[i].parameter(self.N,2*self.N) for k in range(self.N_TV)] for j in range(self.N_modes)])
            self.lmbd_prev.append([[self.opti[i].parameter(4,self.N) for j in range(self.N_modes)] for k in range(self.N_TV)])
            self.nu_prev.append([[self.opti[i].parameter(4,self.N) for j in range(self.N_modes)] for k in range(self.N_TV)])


            self.policy.append(self._return_policy_class(i, N_TV, t_bar))
            self.lmbd_dual_var.append([[self.opti[i].variable(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])
            self.nu_dual_var.append([[self.opti[i].variable(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])
            #             if i==self.t_bar_max-1:
            self._add_constraints_and_cost(i, N_TV, t_bar)
#             self.opti[i].callback(lambda ii: print(self.opti[i].debug.value(self.policy[i][0][0])))
#             self.opti[i].callback(lambda s_iter: print(self.opti[i].debug.value(self.policy[i][0][0])))
#             self._update_ev_initial_condition(i, 0., 10., 0., 0. )
#             self._update_tv_initial_condition(i, N_TV*[25.], N_TV*[10.], N_TV*[0.], N_TV*[self.N_modes*[np.zeros(self.N)]] )
#             self._update_ev_preds(i, self.N_modes*[np.zeros((3,self.N+1))])

#             if i==10:
#                 sol=self.solve(i)


    def _return_policy_class(self, i, N_TV, t_bar):

        if t_bar == 0 or t_bar==self.N-1 or self.OL:
            h=[[self.opti[i].variable(1)] for t in range(self.N)]
            if not self.OL:
                M=[[[self.opti[i].variable(1, 2)] for j in range(t)] for t in range(self.N)]
                K=[[[self.opti[i].variable(1,2) for k in range(N_TV)]] for t in range(self.N)]
            else:
                M=[[[ca.DM(np.zeros((1, 2)))] for j in range(t)] for t in range(self.N)]
                K=[[[ca.DM(np.zeros((1, 2))) for k in range(N_TV)]] for t in range(self.N)]

            M_stack=[ca.vertcat(*[ca.horzcat(*[M[t][j][0] for j in range(t)], ca.MX(1,2*(self.N-t))) for t in range(self.N)])]
            h_stack=[ca.vertcat(*[h[t][0] for t in range(self.N)])]
            K_stack=[[ca.diagcat(*[K[t][0][k] for t in range(self.N)]) for k in range(N_TV)]]
#             pdb.set_trace()
        else:
            h=[[self.opti[i].variable(1) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(self.N)]
#             if not self.OL:
#             M=[[[self.opti[i].variable(1, 2) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for j in range(t)] for t in range(self.N)]
            K=[[[self.opti[i].variable(1,2) for k in range(N_TV)] for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(self.N)]
#             else:
            M=[[[ca.DM.zeros(1, 2) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for j in range(t)] for t in range(self.N)]
#             K=[[[ca.DM(np.zeros((1, 2))) for k in range(N_TV)] for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(self.N)]

            M_stack=[ca.vertcat(*[ca.horzcat(*[M[t][j][m*int(t>=t_bar)] for j in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)]) for m in range(self.N_modes)]
            h_stack=[ca.vertcat(*[h[t][m*int(t>=t_bar)] for t in range(self.N)]) for m in range(self.N_modes)]
            K_stack=[[ca.diagcat(*[K[t][m*int(t>=t_bar)][k] for t in range(self.N)]) for k in range(N_TV)] for m in range(self.N_modes)]

        return h_stack,M_stack,K_stack


    def _get_ATV_TV_dynamics(self, i, N_TV):


            E=ca.DM([[0., 0.],[self.noise_std[2], 0.],[0., self.noise_std[3]]])

            T_tv=[[ca.DM(3*(self.N+1), 3) for j in range(self.N_modes)] for k in range(N_TV)]
            TB_tv=[[ca.DM(3*(self.N+1), self.N) for j in range(self.N_modes)] for k in range(N_TV)]
            c_tv=[[ca.DM(3*(self.N+1), 1) for j in range(self.N_modes)] for k in range(N_TV)]
            E_tv=[[ca.DM(3*(self.N+1),self.N*2) for j in range(self.N_modes)] for k in range(N_TV)]

            u_tvs=self.u_tvs[i]

            elim_stop=False
    #         if y_tv0<0:
    #             elim_stop=True
            c_tv=[]
            for k in range(N_TV):
                c_tvj=[]
                for j in range(self.N_modes):
                    for t in range(self.N+1):
                        if t==0:
                            T_tv[k][j][:3,:]=ca.DM.eye(3)
                        else:
                            T_tv[k][j][t*3:(t+1)*3,:]=self.Atv@T_tv[k][j][(t-1)*3:t*3,:]
                            TB_tv[k][j][t*3:(t+1)*3,:]=self.Atv@TB_tv[k][j][(t-1)*3:t*3,:]
                            TB_tv[k][j][t*3:(t+1)*3,t-1:t]=self.Btv
                            E_tv[k][j][t*3:(t+1)*3,:]=self.Atv@E_tv[k][j][(t-1)*3:t*3,:]
                            E_tv[k][j][t*3:(t+1)*3,(t-1)*2:t*2]=E

                    if k==0:
                        if int(j/2)==0 or elim_stop:
        #                                 if not self.OL:
        #                                     T_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[0])@T_tv[k][j][(t-1)*3:t*3,:]
        #                                     c_tv[k][j][t*3:(t+1)*3,:]=self.Btv@self.gains[0]@ca.DM([0.,-self.o_f, 0.])+(self.Atv-self.Btv@self.gains[0])@c_tv[k][j][(t-1)*3:t*3,:]
        #                                     E_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[0])@E_tv[k][j][(t-1)*3:t*3,:]
        #                                 else:
                            c_tvj.append(TB_tv[k][j]@u_tvs[k][0])
                        else:
        #                                 if not self.OL:
        #                                     T_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[1])@T_tv[k][j][(t-1)*3:t*3,:]
        #                                     c_tv[k][j][t*3:(t+1)*3,:]=self.Btv@self.gains[1]@ca.DM([0.,self.o_s, 0.])+(self.Atv-self.Btv@self.gains[1])@c_tv[k][j][(t-1)*3:t*3,:]
        #                                     E_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[1])@E_tv[k][j][(t-1)*3:t*3,:]
        #                                 else:
                            c_tvj.append(TB_tv[k][j]@u_tvs[k][1])
                    else:
                        if j%2==0:
    #                                 if not self.OL:
    #                                     T_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[0])@T_tv[k][j][(t-1)*3:t*3,:]
    #                                     c_tv[k][j][t*3:(t+1)*3,:]=self.Btv@self.gains[0]@ca.DM([0.,self.o_f, 0.])+(self.Atv-self.Btv@self.gains[0])@c_tv[k][j][(t-1)*3:t*3,:]
    #                                     E_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[0])@E_tv[k][j][(t-1)*3:t*3,:]
    #                                 else:
                            c_tvj.append(TB_tv[k][j]@u_tvs[k][0])
                        else:
    #                                 if not self.OL:
    #                                     T_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[1])@T_tv[k][j][(t-1)*3:t*3,:]
    #                                     c_tv[k][j][t*3:(t+1)*3,:]=self.Btv@self.gains[1]@ca.DM([0.,-self.o_s, 0.])+(self.Atv-self.Btv@self.gains[1])@c_tv[k][j][(t-1)*3:t*3,:]
    #                                     E_tv[k][j][t*3:(t+1)*3,:]=(self.Atv-self.Btv@self.gains[1])@E_tv[k][j][(t-1)*3:t*3,:]
    #                                 else:
                            c_tvj.append(TB_tv[k][j]@u_tvs[k][1])
                c_tv.append(c_tvj)

            return T_tv, c_tv, E_tv



    def _get_LTV_EV_dynamics(self, i, N_TV):


        E=ca.DM([[self.noise_std[0], 0.], [0., 0.], [0., self.noise_std[1]]])

        A_pred=ca.DM(3*(self.N+1), 3)
        B_pred=ca.DM(3*(self.N+1),self.N)
        E_pred=ca.DM(3*(self.N+1),self.N*2)

        A_pred[:3,:]=ca.DM.eye(3)

        for t in range(1,self.N+1):
                A_pred[t*3:(t+1)*3,:]=self.A@A_pred[(t-1)*3:t*3,:]

                B_pred[t*3:(t+1)*3,:]=self.A@B_pred[(t-1)*3:t*3,:]
                B_pred[t*3:(t+1)*3,t-1]=self.B

                E_pred[t*3:(t+1)*3,:]=self.A@E_pred[(t-1)*3:t*3,:]
                E_pred[t*3:(t+1)*3,(t-1)*2:t*2]=E


        return A_pred,B_pred,E_pred

    def _add_constraints_and_cost(self, i, N_TV, t_bar):


        [A,B,E]=self._get_LTV_EV_dynamics(i, N_TV)
        [T_tv,c_tv,E_tv]=self._get_ATV_TV_dynamics(i,N_TV)
#         T_tv=self.T_tv
#         c_tv=self.c_tv
#         E_tv=self.E_tv
        [h,M,K]=self.policy[i]

        nom_z_tv_i=[[T_tv[k][j]@self.z_tv_curr[i][3*k:3*(k+1)]+c_tv[k][j] for k in range(N_TV)] for j in range(self.N_modes)]
        self.nom_z_tv.append(nom_z_tv_i)

        sel_W=ca.kron(ca.DM.eye(self.N), ca.DM([[0.,1.,0],[0., 0., 1.]]))

        cost = 0
        oa_constr_i=[]
        oa_lconstr_i=[]
        for j in range(self.N_modes):
            if len(h)>1:
                uh=h[j]
                h_prev=self.h_prev[i][j]
                uM=M[j]
                M_prev=self.M_prev[i][j]
                uK=K[j]
                K_prev=self.K_prev[i][j]
            else:
                uh=h[0]
                h_prev=self.h_prev[i][0]
                uM=M[0]
                M_prev=self.M_prev[i][0]
                uK=K[0]
                K_prev=self.K_prev[i][0]

            self.opti[i].subject_to(self.opti[i].bounded(self.A_MIN, uh, self.A_MAX))
#             self.opti[i].subject_to(self.opti[i].bounded(-10., ca.diff(uh,1,0), 2.))

            ev_rv=ca.horzcat(B[3:,:]@uM+E[3:,:],*[B[3:,:]@uK[l]@sel_W@E_tv[l][j][3:,:] for l in range(N_TV)])
            oa_constr_t=[]
            oa_lconstr_t=[]
            for t in range(1,self.N+1):

                self.opti[i].subject_to(self.opti[i].bounded(self.V_MIN, A[t*3+2,:]@self.z_curr[i]+B[t*3+2,:]@uh, self.V_MAX))
                oa_constr_k=[]
                oa_lconstr_k=[]
                for k in range(N_TV):
#                     self.opti[i].subject_to(self.opti[i].bounded(-20, ca.vec(uK[k]), 20))

#                     z=self.soc_pred[i][k][j][:-1,t-1].T

#                     tnorm=self.norm_var[i][k][j][:,t-1].T
#                     pdb.set_trace()
#                     y=self.soc_pred[i][k][j][-1,t-1]
                    lmbd=self.lmbd_dual_var[i][k][j][:,t-1].T
                    nu=self.nu_dual_var[i][k][j][:,t-1].T
                    lmbd_prev=self.lmbd_prev[i][k][j][:,t-1].T
                    nu_prev=self.nu_prev[i][k][j][:,t-1].T

#                     pdb.set_trace()
#                         oa_ref=nom_z_tv_i[j][k][3*t:3*t+2]+self.d_nom/ca.norm_2(self.z_lin[i][j][:2,t]-nom_z_tv_i[j][k][3*t:3*t+2])*(self.z_lin[i][j][:2,t]-nom_z_tv_i[j][k][3*t:3*t+2])
#                     oa_ref=nom_z_tv_i[j][k][3*t:3*t+2]+self.d_nom/ca.norm_2(self.z_curr[i][:2]-nom_z_tv_i[j][k][3*t:3*t+2])*(self.z_curr[i][:2]-nom_z_tv_i[j][k][3*t:3*t+2])
                    z_act=(lmbd@self.G+nu@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))
                    y_act=(lmbd@self.G+nu@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])-(lmbd+nu)@self.g
#                     soc_constr=ca.soc(self.tight*(2*(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2]).T@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][:3*self.N,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))),
#                                           -1.*self.d_nom-(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2]).T@(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2])\
#                                           +2*(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2]).T@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:]))
#                     self.opti[i].subject_to(z==(lmbd@self.G+nu@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)])))
#                     self.opti[i].subject_to(y==(lmbd@self.G+nu@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
#                                            -(lmbd+nu)@self.g)
#                     pdb.set_trace()
#                     self.opti[i].subject_to(z==(lmbd_prev@self.G+nu_prev@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))+((lmbd-lmbd_prev)@self.G+(nu-nu_prev)@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@M_prev+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@K_prev[l]@sel_W@E_tv[l][j][3:,:]-float(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)])))
#                     self.opti[i].subject_to(y==(lmbd_prev@self.G+nu_prev@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
#                                             +((lmbd-lmbd_prev)@self.G+(nu-nu_prev)@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@h_prev[j]-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
#                                             -(lmbd+nu)@self.g)
                    z=(lmbd_prev@self.G+nu_prev@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))+((lmbd-lmbd_prev)@self.G+(nu-nu_prev)@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@M_prev+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@K_prev[l]@sel_W@E_tv[l][j][3:,:]-float(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))
#                     z=ca.vertcat(ca.DM.eye(2),self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))
                    y=(lmbd_prev@self.G+nu_prev@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
                      +((lmbd-lmbd_prev)@self.G+(nu-nu_prev)@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@h_prev-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
                      -(lmbd+nu)@self.g
#                     self.opti[i].subject_to(z==0.5*ca.DM.ones(1,4)@(self.G+self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)]))+((lmbd-0.5*ca.DM.ones(1,4))@self.G+(nu-0.5*ca.DM.ones(1,4))@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@M_prev+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@K_prev[l]@sel_W@E_tv[l][j][3:,:]-float(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)])))
#                     self.opti[i].subject_to(y==0.5*ca.DM.ones(1,4)@(self.G+self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
#                                             +((lmbd-0.5*ca.DM.ones(1,4))@self.G+(nu-0.5*ca.DM.ones(1,4))@self.G@self.Rtv.T)@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@h_prev[j]-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:])\
#                                             -(lmbd+nu-ca.DM.ones(1,4))@self.g)

                    self.opti[i].subject_to(lmbd>=0)
                    self.opti[i].subject_to(nu>=0)
                    self.opti[i].subject_to(ca.norm_2(ca.horzcat((lmbd)@self.G,(nu)@self.G))<=1)
#                     self.opti[i].subject_to(soc_constr>0)
#                     self.opti[i].subject_to(z==2*(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2]).T@ca.DM([[1, 0, 0], [0, 1, 0]])@(ca.horzcat(B[t*3:(t+1)*3,:]@uM+E[t*3:(t+1)*3,:],*[B[t*3:(t+1)*3,:]@uK[l]@sel_W@E_tv[l][j][3:,:]-int(l==k)*E_tv[k][j][t*3:(t+1)*3,:] for l in range(N_TV)])))
#                     self.opti[i].subject_to(y==-1.*self.d_nom-(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2]).T@(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2])\
#                         +2*(oa_ref-nom_z_tv_i[j][k][3*t:3*t+2]).T@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr[i]+B[t*3:(t+1)*3,:]@uh-T_tv[k][j][t*3:(t+1)*3,:]@self.z_tv_curr[i][3*k:3*(k+1)]-c_tv[k][j][t*3:(t+1)*3,:]))
                    self.opti[i].subject_to(ca.norm_2(self.tight*z)<=y-self.d_nom)
                    oa_lconstr_k.append(y-ca.norm_2(self.tight*z))
                    oa_constr_k.append(y_act-ca.norm_2(self.tight*z_act))
#                     self.opti[i].subject_to(self.opti[i].bounded(-tnorm,self.tight*z,tnorm))
#                     self.opti[i].subject_to(tnorm<=y)
                oa_constr_t.append(oa_constr_k)
                oa_lconstr_t.append(oa_lconstr_k)
            oa_constr_i.append(oa_constr_t)
            oa_lconstr_i.append(oa_lconstr_t)
#             cost+=(A[t*3:(t+1)*3:2,:]@self.z_curr[i]+B[t*3:(t+1)*3:2,:]@uh-ca.DM([self.s_f, 0.])).T@self.Q@(A[t*3:(t+1)*3:2,:]@self.z_curr[i]+B[t*3:(t+1)*3:2,:]@uh-ca.DM([self.s_f, 0.]))
#             cost+=ca.diff(A[t*3:(t+1)*3:2,:]@self.z_curr[i]+B[t*3:(t+1)*3:2,:]@uh
#                     cost+=-self.Q[0,0]*(A[t*3,:]@self.z_curr[i]+B[t*3,:]@h[j])

#             cost+=uh[t-1]@self.R@uh[t-1]
#                 soc_term=ca.soc(ca.norm_2(ca.vertcat(0.5+1*self.A_MIN*20-1*self.A_MIN*(A[3*self.N,:]@self.z_curr[i]+B[3*self.N,:]@h[j]),A[3*self.N+2,:]@self.z_curr[i]+B[3*self.N+2,:]@h[j])),
#                                 0.5-1*self.A_MIN*(20-A[3*self.N,:]@self.z_curr[i]+B[3*self.N,:]@h[j]))
#                 self.opti[i].subject_to(soc_term>0)
#                 if self.term_flag[i]>0.:
#                     self.opti[i].subject_to(ca.norm_2(ca.vertcat(0.5+1*self.A_MIN*20-1*self.A_MIN*(A[3*self.N,:]@self.z_curr[i]+B[3*self.N,:]@h[j]),A[3*self.N+2,:]@self.z_curr[i]+B[3*self.N+2,:]@h[j]))<=0.5-1*self.A_MIN*(20-A[3*self.N,:]@self.z_curr[i]-B[3*self.N,:]@h[j]))

            nom_z=A@self.z_curr[i]+B@uh
            nom_z_diff=ca.diff(nom_z.reshape((3,-1)),1,1).reshape((-1,1))
#             cost+=1*(nom_z_diff.T@nom_z_diff+1.*ca.trace(ev_rv@ev_rv.T))-self.Q*ca.sum1(ca.diff(nom_z.reshape((3,-1)),1,1)[0,:].T)
            cost+=-self.Q*ca.sum1(ca.diff(nom_z.reshape((3,-1)),1,1)[0,:].T)
            cost+=self.R*ca.diff(ca.vertcat(self.u_prev[i],uh),1,0).T@ca.diff(ca.vertcat(self.u_prev[i],uh),1,0)


        self.opti[i].minimize( cost )
        self.oa_constr.append(oa_constr_i)
        self.oa_lconstr.append(oa_lconstr_i)


    def solve(self, i):
        st = time.time()

        try:
#             pdb.set_trace()
            sol = self.opti[i].solve()
            # Optimal solution.
#             pdb.set_trace()
            u_control  = sol.value(self.policy[i][0][0][0])
            if not (self.OL or i==self.N-1):
                h_opt      = [sol.value(self.policy[i][0][j]) for j in range(self.N_modes)]
                M_opt      = [sol.value(self.policy[i][1][j]) for j in range(self.N_modes)]
                K_opt      = [[sol.value(self.policy[i][2][j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
                lmbd_opt    = [[sol.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(self.N_TV)]
                nu_opt     = [[sol.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(self.N_TV)]
            else:
                h_opt      = [sol.value(self.policy[i][0][0]) for j in range(self.N_modes)]
                M_opt      = [sol.value(self.policy[i][1][0]) for j in range(self.N_modes)]
                K_opt      = [[sol.value(self.policy[i][2][0][k]) for k in range(len(self.policy[i][2][0]))] for j in range(self.N_modes)]
                lmbd_opt    = [[sol.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(self.N_TV)]
                nu_opt     = [[sol.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(self.N_TV)]

            nom_z_tv   = [[sol.value(self.nom_z_tv[i][j][k]) for k in range(len(self.policy[i][2][0]))] for j in range(self.N_modes)]

            is_opt     = True
        except:
            # Suboptimal solution (e.g. timed out).

            subsol=self.opti[i].debug
            pdb.set_trace()
            u_control  = self.u_backup
            if not np.any(np.isnan(subsol.value(self.policy[i][0][0][0]))):
                u_control=subsol.value(self.policy[i][0][0][0])
            if not (self.OL or i==self.N-1):
                h_opt      = [subsol.value(self.policy[i][0][j]) for j in range(self.N_modes)]
                M_opt      = [subsol.value(self.policy[i][1][j]) for j in range(self.N_modes)]
                K_opt      = [[subsol.value(self.policy[i][2][j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
                lmbd_opt    = [[subsol.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(self.N_TV)]
                nu_opt     = [[subsol.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(self.N_TV)]
            else:
                h_opt      = [subsol.value(self.policy[i][0][0]) for j in range(self.N_modes)]
                M_opt      = [subsol.value(self.policy[i][1][0]) for j in range(self.N_modes)]
                K_opt      = [[subsol.value(self.policy[i][2][0][k]) for k in range(len(self.policy[i][2][0]))] for j in range(self.N_modes)]
                lmbd_opt    = [[subsol.value(self.lmbd_dual_var[i][k][0]) for j in range(self.N_modes)] for k in range(self.N_TV)]
                nu_opt     = [[subsol.value(self.nu_dual_var[i][k][0]) for j in range(self.N_modes)] for k in range(self.N_TV)]

            nom_z_tv   = [[subsol.value(self.nom_z_tv[i][j][k]) for k in range(len(self.policy[i][2][0]))] for j in range(self.N_modes)]


            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        if is_opt:
            sol_dict['h_opt']=h_opt
            sol_dict['M_opt']=M_opt
            sol_dict['K_opt']=K_opt
            sol_dict['lmbd_opt']=lmbd_opt
            sol_dict['nu_opt']=nu_opt
        sol_dict['nom_z_tv']=nom_z_tv

        sol_dict['solve_time'] = solve_time  # how long the solver took in seconds


        return sol_dict

    def update(self, i, update_dict):
        self._update_ev_initial_condition(i, *[update_dict[key] for key in ['x0','y0', 'v0', 'u_prev']] )
        self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0', 'v_tv0', 'u_tvs']] )
        self._update_ev_preds(i, update_dict['x_lin'])


        for j in range(self.N_modes):
            if j<len(self.policy[i][0]):
                self.opti[i].set_value(self.h_prev[i][j],self.u_backup*ca.DM.ones(self.N,1))
                self.opti[i].set_value(self.M_prev[i][j],ca.DM.zeros(self.N,2*self.N))
            if 'ws' in update_dict.keys():
                if j<len(self.policy[i][0]):
#                     self.opti[i].set_initial(self.policy[i][0][j],  update_dict['ws'][0][j])
                    self.opti[i].set_initial(self.policy[i][0][j],  self.u_backup*ca.DM.ones(self.N,1))
                    self.opti[i].set_initial(self.policy[i][1][j],  update_dict['ws'][1][j])
                    self.opti[i].set_value(self.h_prev[i][j],update_dict['ws'][0][j])
                    self.opti[i].set_value(self.M_prev[i][j],update_dict['ws'][1][j])

            for k in range(self.N_TV):

                self.opti[i].set_initial(self.lmbd_dual_var[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_initial(self.nu_dual_var[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_value(self.lmbd_prev[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_value(self.nu_prev[i][k][j],  0.5*ca.DM.ones(4,self.N))
                if j<len(self.policy[i][0]):
                    self.opti[i].set_value(self.K_prev[i][j][k],ca.DM.zeros(self.N,2*self.N))

                if 'ws' in update_dict.keys():
                    if j<len(self.policy[i][0]):
                        self.opti[i].set_initial(self.policy[i][2][j][k], update_dict['ws'][2][j][k])
                        self.opti[i].set_value(self.K_prev[i][j][k], update_dict['ws'][2][j][k])
                    self.opti[i].set_initial(self.lmbd_dual_var[i][k][j], update_dict['ws'][3][k][j])
                    self.opti[i].set_initial(self.nu_dual_var[i][k][j], update_dict['ws'][4][k][j])

                    self.opti[i].set_value(self.lmbd_prev[i][k][j], update_dict['ws'][3][k][j])
                    self.opti[i].set_value(self.nu_prev[i][k][j], update_dict['ws'][4][k][j])


    def _update_ev_initial_condition(self, i, x0, y0,  v0, u_prev):
        self.opti[i].set_value(self.z_curr[i], ca.DM([x0, y0, v0]))
        self.opti[i].set_value(self.u_prev[i], u_prev)
#         if self.OL:
#             self.u_backup=u_prev
#         else:
#         self.u_backup=self.A_MIN
        self.u_backup=self.A_MIN

    def _update_tv_initial_condition(self, i, x_tv0, y_tv0, v_tv0, u_tvs):


        for k in range(self.N_TV):
            self.opti[i].set_value(self.z_tv_curr[i][3*k:3*(k+1)], ca.DM([x_tv0[k], y_tv0[k], v_tv0[k]]))
            for j in range(self.N_modes):
                self.opti[i].set_value(self.u_tvs[i][k][j], u_tvs[k][j])
#             if y_tv0[0]>0.:
#                 self.opti[i].set_value(self.term_flag[i], 1.)
#             else:
#                 self.opti[i].set_value(self.term_flag[i], 0.)



    def _update_ev_preds(self, i, x_lin):

        for j in range(self.N_modes):

            self.opti[i].set_value(self.z_lin[i][j],x_lin[j])

