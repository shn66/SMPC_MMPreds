import time
import casadi as ca
import numpy as np
from itertools import product
import scipy.stats as stats
from scipy.stats import norm
import pdb
class RefTrajGenerator():

    def __init__(self,
                 N          = 100,     # timesteps in Optimization Horizon
                 DT         = 0.2,    # discretization time between timesteps (s)
                 L_F        = 1.7213, # distance from CoG to front axle (m)
                 L_R        = 1.4987, # distance from CoG to rear axle (m)
                 V_MIN        = 0.,
                 V_MAX        = 20.0,
                 A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
                 A_MAX      =  2.0,
                 DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
                 DF_MAX     =  0.5,
                 A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
                 A_DOT_MAX  =  1.5,
                 DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
                 DF_DOT_MAX =  0.5,
                 Q = [10., 10., 500., 0.1], # weights on x, y, and v.
                 R = [0.0001*10, 0.0001*100.]):        # weights on inputs

        for key in list(locals()):
            if key == 'self':
                pass
            elif key == 'Q':
                self.Q = ca.diag(Q)
            elif key == 'R':
                self.R = ca.diag(R)
            else:
                setattr(self, '%s' % key, locals()[key])

        self.opti = ca.Opti()

        '''
        (1) Parameters
        '''
        self.u_prev  = self.opti.parameter(2) # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_curr  = self.opti.parameter(4) # current state:  [x_0, y_0, psi_0, v_0]

        # Waypoints we would like to follow.

        self.wp_ref = self.opti.parameter(self.N, 4)

        self.x_wp   = self.wp_ref[:,0]
        self.y_wp   = self.wp_ref[:,1]
        self.psi_wp   = self.wp_ref[:,2]
        self.v_wp   = self.wp_ref[:,3]

        '''
        (2) Decision Variables
        '''
        # Actual trajectory we will follow given the optimal solution.
        # First index is the timestep k, i.e. self.z_dv[0,:] is z_0.
        # It has self.N+1 timesteps since we go from z_0, ..., z_self.N.
        # Second index is the state element, as detailed below.
        self.z_dv = self.opti.variable(self.N+1, 4)

        self.x_dv   = self.z_dv[:, 0]
        self.y_dv   = self.z_dv[:, 1]
        self.psi_dv = self.z_dv[:, 2]
        self.v_dv   = self.z_dv[:, 3]

        # Control inputs used to achieve self.z_dv according to dynamics.
        # First index is the timestep k, i.e. self.u_dv[0,:] is u_0.
        # Second index is the input element as detailed below.
        self.u_dv = self.opti.variable(self.N, 2)

        self.acc_dv = self.u_dv[:,0]
        self.df_dv  = self.u_dv[:,1]

#       # Slack variables used to relax input rate constraints.
#       # Matches self.u_dv in structure but timesteps range from -1, ..., N-1.
        self.sl_dv  = self.opti.variable(self.N , 2)

        self.sl_acc_dv = self.sl_dv[:,0]
        self.sl_df_dv  = self.sl_dv[:,1]

        '''
        (3) Problem Setup: Constraints, Cost, Initial Solve
        '''
        self._add_constraints()

        self._add_cost()

        ## Uncomment for initial solve (Required only if this object is repeatedly re-initialised and solved at high frequency)

        # Ipopt with custom options: https://web.ca.org/docs/ -> see sec 9.1 on Opti stack.
        p_opts = {'expand': True}
        s_opts = {'max_cpu_time': 0.25, 'print_level': 0}
        self.opti.solver('ipopt', p_opts, s_opts)

        # sol = self.solve()

    def _add_constraints(self):
        # State Bound Constraints
        self.opti.subject_to( self.opti.bounded(self.V_MIN, self.v_dv, self.V_MAX) )

        # Initial State Constraint
        self.opti.subject_to( self.x_dv[0]   == self.z_curr[0] )
        self.opti.subject_to( self.y_dv[0]   == self.z_curr[1] )
        self.opti.subject_to( self.psi_dv[0] == self.z_curr[2] )
        self.opti.subject_to( self.v_dv[0]   == self.z_curr[3] )

        # State Dynamics Constraints
        for i in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_dv[i]) )
            self.opti.subject_to( self.x_dv[i+1]   == self.x_dv[i]   + self.DT * (self.v_dv[i] * ca.cos(self.psi_dv[i] + beta)) )
            self.opti.subject_to( self.y_dv[i+1]   == self.y_dv[i]   + self.DT * (self.v_dv[i] * ca.sin(self.psi_dv[i] + beta)) )
            self.opti.subject_to( self.psi_dv[i+1] == self.psi_dv[i] + self.DT * (self.v_dv[i] / self.L_R * ca.sin(beta)) )
            self.opti.subject_to( self.v_dv[i+1]   == self.v_dv[i]   + self.DT * (self.acc_dv[i]) )


        # Input Bound Constraints
        self.opti.subject_to( self.opti.bounded(self.A_MIN,  self.acc_dv, self.A_MAX) )
        self.opti.subject_to( self.opti.bounded(self.DF_MIN, self.df_dv,  self.DF_MAX) )
      # Input Rate Bound Constraints
        self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT -  self.sl_acc_dv[0],
                                               self.acc_dv[0] - self.u_prev[0],
                                               self.A_DOT_MAX*self.DT   + self.sl_acc_dv[0]) )

        self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[0],
                                               self.df_dv[0] - self.u_prev[1],
                                               self.DF_DOT_MAX*self.DT  + self.sl_df_dv[0]) )

        for i in range(self.N - 1):
            self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT   -  self.sl_acc_dv[i+1],
                                                   self.acc_dv[i+1] - self.acc_dv[i],
                                                   self.A_DOT_MAX*self.DT   + self.sl_acc_dv[i+1]) )
            self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[i+1],
                                                   self.df_dv[i+1]  - self.df_dv[i],
                                                   self.DF_DOT_MAX*self.DT  + self.sl_df_dv[i+1]) )
        # Other Constraints
        self.opti.subject_to( 0 <= self.sl_df_dv )
        self.opti.subject_to( 0 <= self.sl_acc_dv )
        # e.g. things like collision avoidance or lateral acceleration bounds could go here.
    @staticmethod
    def _quad_form(z, Q):
            return ca.mtimes(z.T, ca.mtimes(Q, z))

    def _add_cost(self):
        cost = 0
        for i in range(self.N):
            cost += RefTrajGenerator._quad_form(self.z_dv[i+1, :].T - self.wp_ref[i,:].T, self.Q) # tracking cost

        for i in range(self.N-1 ):
            cost += RefTrajGenerator._quad_form(self.u_dv[i+1,:].T-self.u_dv[i,:].T, self.R)  # input cost

        cost += (ca.sum1(self.sl_df_dv) + ca.sum1(self.sl_acc_dv))  # slack cost

        self.opti.minimize( cost )

    def solve(self):
        st = time.time()
        try:
            sol = self.opti.solve()
            # Optimal solution.
            u_opt  = sol.value(self.u_dv)
            z_opt  = sol.value(self.z_dv)
#           sl_mpc = sol.value(self.sl_dv)
            wp_ref  = sol.value(self.wp_ref)
            is_opt = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_opt  = self.opti.debug.value(self.u_dv)
            z_opt  = self.opti.debug.value(self.z_dv)
#           sl_mpc = self.opti.debug.value(self.sl_dv)
            wp_ref  = self.opti.debug.value(self.wp_ref)
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        # sol_dict['u_control']  = u_opt[0,:]  # control input to apply based on solution
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        sol_dict['solve_time'] = solve_time  # how long the solver took in seconds
        sol_dict['u_opt']      = u_opt       # solution inputs (N by 2, see self.u_dv above)
        sol_dict['z_opt']      = z_opt       # solution states (N+1 by 4, see self.z_dv above)
#       sol_dict['sl_mpc']     = sl_mpc      # solution slack vars (N by 2, see self.sl_dv above)
        sol_dict['wp_ref']     = wp_ref      # waypoints  (N by 4, see self.wp_ref above)

        return sol_dict

    def update(self, update_dict):
        self._update_initial_condition( *[update_dict[key] for key in ['x0', 'y0', 'psi0', 'v0']] )
        self._update_reference( *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref']] )
        self._update_previous_input( *[update_dict[key] for key in ['acc_prev', 'df_prev']] )

        if 'warm_start' in update_dict.keys():
            # Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
            self.opti.set_initial(self.z_dv,  update_dict['warm_start']['z_ws'])
            self.opti.set_initial(self.u_dv,  update_dict['warm_start']['u_ws'])
#           self.opti.set_initial(self.sl_dv, update_dict['warm_start']['sl_ws'])

    def _update_initial_condition(self, x0, y0, psi0, vel0):
        self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0])

    def _update_reference(self, x_ref, y_ref, psi_ref, v_ref):
        self.opti.set_value(self.x_wp,   x_ref)
        self.opti.set_value(self.y_wp,   y_ref)
        self.opti.set_value(self.psi_wp, psi_ref)
        self.opti.set_value(self.v_wp,   v_ref)


    def _update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])



# class SMPC_MMPreds():

#     def __init__(self,
#                 N            = 10,
#                 DT           = 0.2,
#                 L_F          = 1.7213,
#                 L_R          = 1.4987,
#                 V_MIN        = 0.0,
#                 V_MAX        = 20.0,
#                 A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
#                 A_MAX      =  2.0,
#                 DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
#                 DF_MAX     =  0.5,
#                 A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
#                 A_DOT_MAX  =  1.5,
#                 DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
#                 DF_DOT_MAX =  0.5,
#                 N_modes_MAX  =  3,
#                 N_TV_MAX     =  1,
#                 N_seq_MAX    =  100,
#                 T_BAR_MAX    =  6,
#                 TIGHTENING   =  1.64,
#                 NOISE_STD    =  [0.1, .1, .01, .1, 0.1], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
#                 Q =[0.1*50., 0.001*50, 1*10., 0.1*10.], # weights on x, y, and v.
#                 R = [10., 1000],       # weights on inputs
#                 # NOISE_STD    =  [0.1, 0.1, .2, 0.5, 0.5],
#                 # Q =[.050, .050, 10, 0.1], # weights on x, y, and v.
#                 # R = [.1*10, .1*100],
#                 NS_BL_FLAG=False,
#                 fps = 20
#                 ):
#         self.N=N
#         self.DT=DT
#         self.L_F=L_F
#         self.L_R=L_R
#         self.V_MIN=V_MIN
#         self.V_MAX=V_MAX
#         self.A_MIN=A_MIN
#         self.A_MAX=A_MAX
#         self.DF_MIN=DF_MIN
#         self.DF_MAX=DF_MAX
#         self.A_DOT_MIN=A_DOT_MIN
#         self.A_DOT_MAX=A_DOT_MAX
#         self.DF_DOT_MIN=DF_DOT_MIN
#         self.DF_DOT_MAX=DF_DOT_MAX
#         self.N_modes=N_modes_MAX
#         self.N_TV_max=N_TV_MAX
#         self.N_seq_max=N_seq_MAX
#         self.t_bar_max=T_BAR_MAX
#         self.tight=TIGHTENING
#         self.noise_std=NOISE_STD
#         self.Q = ca.diag(Q)
#         self.R = ca.diag(R)
#         self.a_brake=-7.0
#         self.v_curr=0.0
#         self.noswitch_bl=NS_BL_FLAG
#         self.fps=fps

#         self.opti=[]

#         self.z_ref=[]
#         self.z_lin=[]
#         self.u_prev=[]
#         self.x_ref=[]
#         self.y_ref=[]
#         self.psi_ref=[]
#         self.v_ref=[]
#         self.u_ref=[]
#         self.a_ref=[]
#         self.df_ref=[]
#         self.x_lin=[]
#         self.y_lin=[]
#         self.psi_lin=[]
#         self.v_lin=[]
#         self.u_lin=[]
#         self.a_lin=[]
#         self.df_lin=[]
#         self.dz_curr=[]
#         self.Sigma_tv_sqrt  =  []
#         self.Q_tv = []
# #         self.Mu_tv=[]
# #         self.Sigma_tv=[]
#         self.T_tv=[]
#         self.c_tv=[]
#         # self.z_tv_ref=[]
#         self.x_tv_ref=[]
#         self.y_tv_ref=[]
#         self.z_tv_curr=[]
#         self.rot_costs=[]

#         self.policy=[]
#         self.slacks=[]

#         self.nom_z_ev = []
#         self.nom_u_ev = []
#         self.eval_oa=[]


#         p_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-3, 'PSDTol' : 1e-3}
#         s_opts_grb = {'error_on_fail':0}


#         for i in range((self.t_bar_max)*self.N_TV_max):
#             self.opti.append(ca.Opti('conic'))
#             self.opti[i].solver("gurobi", s_opts_grb, p_opts_grb)
#             # self.opti[i].solver("superscs", s_opts_grb, {})

#             N_TV=1+int(i/self.t_bar_max)
#             t_bar=i-(N_TV-1)*self.t_bar_max

#             self.z_ref.append(self.opti[i].parameter(4, self.N+1))
#             self.u_ref.append(self.opti[i].parameter(2, self.N+1))
#             self.u_prev.append(self.opti[i].parameter(2))
#             self.x_ref.append(self.z_ref[i][0, :])
#             self.y_ref.append(self.z_ref[i][1, :] )
#             self.psi_ref.append(self.z_ref[i][2, :] )
#             self.v_ref.append(self.z_ref[i][3, :])

#             self.a_ref.append(self.u_ref[i][0, :])
#             self.df_ref.append(self.u_ref[i][1, :])

#             self.z_lin.append(self.opti[i].parameter(4, self.N+1))
#             self.u_lin.append(self.opti[i].parameter(2, self.N+1))
#             self.x_lin.append(self.z_lin[i][0, :])
#             self.y_lin.append(self.z_lin[i][1, :] )
#             self.psi_lin.append(self.z_lin[i][2, :] )
#             self.v_lin.append(self.z_lin[i][3, :])

#             self.a_lin.append(self.u_lin[i][0, :])
#             self.df_lin.append(self.u_lin[i][1, :])

#             self.dz_curr.append(self.opti[i].parameter(4))
#             self.slacks.append(self.opti[i].variable(1))



# #             self.Mu_tv.append([[self.opti[i].parameter(2, self.N) for j in range(self.N_modes)] for k in range(N_TV)])
# #             self.Sigma_tv.append([[[self.opti[i].parameter(2, 2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])

#             self.T_tv.append([[[self.opti[i].parameter(2,2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])
#             self.c_tv.append([[[self.opti[i].parameter(2,1) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])

#             # self.z_tv_ref.append([[self.opti[i].parameter(2,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])


#             self.x_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])
#             self.y_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])

#             self.Sigma_tv_sqrt.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])
#             self.Q_tv.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])

#             self.z_tv_curr.append(self.opti[i].parameter(2,N_TV))
#             self.rot_costs.append([self.opti[i].parameter(4,4) for t in range(self.N)])
#             self.policy.append(self._return_policy_class(i, N_TV, t_bar))
#             self._add_constraints_and_cost(i, N_TV, t_bar)
#             self.u_ref_val=np.zeros((2,1))
#             self.v_next=np.array(5.)
#             self._update_ev_initial_condition(i, 0., 0., np.pi*0., 5.0 )
#             self._update_ev_rotated_costs(i, self.N*[np.identity(2)])
#             self._update_tv_initial_condition(i, N_TV*[20.0], N_TV*[20.0] )
#             self._update_ev_reference(i, [self.DT *5.0* (x) for x in range(self.N+1)],
#                                       [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
#             self._update_ev_lin(i, [self.DT *5.0* (x) for x in range(self.N+1)],
#                                       [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
#             self._update_tv_preds(i, N_TV*[20.0], N_TV*[20.0], N_TV*[20*np.ones((self.N_modes, self.N, 2))], N_TV*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
#             self._update_previous_input(i, 0.0, 0.0)
#             self._update_tv_shapes(i, N_TV*[self.N_modes*[self.N*[0.1*np.identity(2)]]])
#             sol=self.solve(i)


#     # def _return_policy_class(self, i, N_TV, t_bar):

#     #     if t_bar == 0 or t_bar==self.N-1:
#     #         M=[self.opti[i].variable(2, 4+2*N_TV) for j in range(int((self.N-1)*self.N/2))]
#     #         h=self.opti[i].variable(2, self.N)

#     #     else:
#     #         M=[self.opti[i].variable(2, 4+2*N_TV) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes**N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
#     #         h=self.opti[i].variable(2, t_bar+(self.N_modes**N_TV)*(self.N-t_bar))

#     #     return h,M

#     def _return_policy_class(self, i, N_TV, t_bar):

#         if t_bar == 0 or t_bar==self.N-1:
#             M=[self.opti[i].variable(2, 4) for j in range(int((self.N-1)*self.N/2))]
#             K=[self.opti[i].variable(2,2*N_TV) for j in range(self.N)]
#             h=self.opti[i].variable(2, self.N)

#         else:
#             M=[self.opti[i].variable(2, 4) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes**N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
#             # K=[self.opti[i].variable(2,2*N_TV) for j in range(self.N)]
#             K=[self.opti[i].variable(2,2*N_TV) for j in range(t_bar+(self.N_modes**N_TV)*(self.N-t_bar))]
#             h=self.opti[i].variable(2, t_bar+(self.N_modes**N_TV)*(self.N-t_bar))

#         return h,K,M

#     def _set_ATV_TV_dynamics(self, i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv):


#         T=self.T_tv[i]
#         c=self.c_tv[i]

#         for k in range(N_TV):
#             for j in range(self.N_modes):
#                 for t in range(self.N):
#                     if t==0:
#                         self.opti[i].set_value(T[k][j][t], np.identity(2))
#                         self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-np.hstack((x_tv0[k],y_tv0[k])))
#                         e_val,e_vec= np.linalg.eigh(sigma_tv[k][j,t,:,:])
#                         self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], e_vec@np.diag(np.sqrt(e_val))@e_vec.T )
#                     else:

#                         e_val,e_vec= np.linalg.eigh(sigma_tv[k][j,t,:,:])
#                         e_valp,e_vecp= np.linalg.eigh(sigma_tv[k][j,t-1,:,:])
#                         Ttv=e_vec@np.diag(np.sqrt(e_val))@e_vec.T@e_vecp@np.diag(np.sqrt(e_valp)**(-1))@e_vecp.T
#                         # Ltp1=ca.chol(sigma_tv[k][j,t,:,:])
#                         # Lt=ca.chol(sigma_tv[k][j,t-1,:,:])
#                         self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], ca.chol(sigma_tv[k][j,t,:,:]) )
#                         self.opti[i].set_value(T[k][j][t], Ttv)
#                         # self.opti[i].set_value(T[k][j][t], np.identity(2))
#                         self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-Ttv@mu_tv[k][j, t-1, :])



#     def _set_TV_ref(self, i, N_TV, x_tv0, y_tv0, mu_tv):

#         for k in range(N_TV):
#             for j in range(self.N_modes):
#                 for t in range(self.N+1):
#                     if t==0:
#                         self.opti[i].set_value(self.x_tv_ref[i][k][j][0], x_tv0[k])
#                         self.opti[i].set_value(self.y_tv_ref[i][k][j][0], y_tv0[k])
#                     else:
#                         self.opti[i].set_value(self.x_tv_ref[i][k][j][t], mu_tv[k][j,t-1,0])
#                         self.opti[i].set_value(self.y_tv_ref[i][k][j][t], mu_tv[k][j,t-1,1])




#     def _get_LTV_EV_dynamics(self, i, N_TV):

#         A=[ca.MX.eye(4) for n in range(self.N+1)]
#         B=[ca.MX(4, 2) for n in range(self.N+1)]

#         for t in range(self.N):
#             beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]) )
#             dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_lin[i][t])**2
#             # beta=self.L_R / (self.L_F + self.L_R)*self.df_lin[i][t]
#             # dbeta=self.L_R / (self.L_F + self.L_R)

#             # pdb.set_trace()
#             # At=ca.MX.eye(4)
#             # Atp1=ca.MX.eye(4)
#             # At[0,2]+=0.5*self.DT*(-ca.fmax(self.v_lin[i][t],0.1)*ca.sin(self.psi_lin[i][t]+beta))
#             # At[0,3]+=0.5*self.DT*(ca.cos(self.psi_lin[i][t]+beta))
#             # At[1,2]+=0.5*self.DT*(ca.fmax(self.v_lin[i][t],0.1)*ca.cos(self.psi_lin[i][t]+beta))
#             # At[1,3]+=0.5*self.DT*(ca.sin(self.psi_lin[i][t]+beta))
#             # At[2,3]+=0.5*self.DT*(1./self.L_R*ca.sin(beta))

#             # Atp1[0,2]-=0.5*self.DT*(-ca.fmax(self.v_lin[i][t+1],0.1)*ca.sin(self.psi_lin[i][t+1]+dbeta*self.df_lin[i][t+1]))
#             # Atp1[0,3]-=0.5*self.DT*(ca.cos(self.psi_lin[i][t+1]+dbeta*self.df_lin[i][t+1]))
#             # Atp1[1,2]-=0.5*self.DT*(ca.fmax(self.v_lin[i][t+1],0.1)*ca.cos(self.psi_lin[i][t+1]+dbeta*self.df_lin[i][t+1]))
#             # Atp1[1,3]-=0.5*self.DT*(ca.sin(self.psi_lin[i][t+1]+dbeta*self.df_lin[i][t+1]))
#             # Atp1[2,3]-=0.5*self.DT*(1./self.L_R*ca.sin(dbeta*self.df_lin[i][t+1]))


#             # A[t]=ca.inv(Atp1)@At


#             A[t][0,2]+=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta))
#             A[t][0,3]+=self.DT*(ca.cos(self.psi_lin[i][t]+beta))
#             A[t][1,2]+=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta))
#             A[t][1,3]+=self.DT*(ca.sin(self.psi_lin[i][t]+beta))
#             A[t][2,3]+=self.DT*(1./self.L_R*ca.sin(beta))
#             # A[t]=ca.expm(A[t])

#             # Bt=ca.MX(4,2)
#             # Btp1=ca.MX(4,2)

#             # Bt[0,1]=0.5*self.DT*(-ca.fmax(self.v_lin[i][t],0.1)*ca.sin(self.psi_lin[i][t]+beta)*dbeta)
#             # Bt[1,1]=0.5*self.DT*(ca.fmax(self.v_lin[i][t],0.1)*ca.cos(self.psi_lin[i][t]+beta)*dbeta)
#             # Bt[2,1]=0.5*self.DT*(ca.fmax(self.v_lin[i][t],0.1)/self.L_R*ca.cos(beta)*dbeta)
#             # Bt[3,0]=0.5*self.DT*1.0

#             # Btp1[0,1]=0.5*self.DT*(-ca.fmax(self.v_lin[i][t+1],0.1)*ca.sin(self.psi_lin[i][t+1]+dbeta*self.df_lin[i][t+1])*dbeta)
#             # Btp1[1,1]=0.5*self.DT*(ca.fmax(self.v_lin[i][t+1],0.1)*ca.cos(self.psi_lin[i][t+1]+dbeta*self.df_lin[i][t+1])*dbeta)
#             # Btp1[2,1]=0.5*self.DT*(ca.fmax(self.v_lin[i][t+1],0.1)/self.L_R*ca.cos(dbeta*self.df_lin[i][t+1])*dbeta)
#             # Btp1[3,0]=0.5*self.DT*1.0

#             # B[t]=ca.inv(Atp1)@(Bt+Btp1)



#             B[t][0,1]=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta)*dbeta)
#             B[t][1,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta)*dbeta)
#             B[t][2,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)/self.L_R*ca.cos(beta)*dbeta)
#             B[t][3,0]=self.DT*1.0


#         E=ca.MX(4+2*N_TV, 4+2*N_TV)
#         # E=(ca.MX.eye(4))@ca.diag(self.noise_std[0:4])
#         E[0:4,0:4]=(ca.MX.eye(4))@ca.diag(self.noise_std[0:4])
#         E[4:, 4:]=ca.MX.eye(2*N_TV)*self.noise_std[-1]

#         return A,B,E

#     def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv, Q):
#             x_ev_avg=0.5*(x_ev[0]+x_ev[1])
#             y_ev_avg=0.5*(y_ev[0]+y_ev[1])
#             # x_ev_avg=x_ev[1]
#             # y_ev_avg=y_ev[1]


#             x_ref_ev=x_tv+(x_ev_avg-x_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
#             y_ref_ev=y_tv+(y_ev_avg-y_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
#             # x_ref_ev=x_tv+self.d_min*((x_ev[1])-x_tv)/ca.norm_2(ca.vertcat(x_ev[1],y_ev[1])-ca.vertcat(x_tv,y_tv))
#             # y_ref_ev=y_tv+self.d_min*((y_ev[1])-y_tv)/ca.norm_2(ca.vertcat(x_ev[1],y_ev[1])-ca.vertcat(x_tv,y_tv))
#             # pdb.set_trace()
#             return ca.vertcat(x_ref_ev,y_ref_ev)




#     def _add_constraints_and_cost(self, i, N_TV, t_bar):


#         nom_z_ev_i = []
#         nom_u_ev_i = []
#         eval_oa_i=[]

#         T=self.T_tv[i]
#         c=self.c_tv[i]
#         [A,B,E]=self._get_LTV_EV_dynamics(i, N_TV)
#         [h,K,M]=self.policy[i]
#         slack=self.slacks[i]
#         cost = 1*slack@slack
#         x=0.5*(self.dz_curr[i][0]+self.x_lin[i][0]+self.x_lin[i][-1])
#         y=0.5*(self.dz_curr[i][0]+self.y_lin[i][0]+self.y_lin[i][-1])
#         self.opti[i].subject_to(slack>=0)
#         self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
#                                                       self.v_lin[i][1]+A[0][3,:]@self.dz_curr[i]+B[0][3,:]@h[:,0],
#                                                       self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )

#         self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,self.a_lin[i][0]+h[0,0],self.A_MAX))
#         # self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN,h[0,0]*self.fps,self.A_DOT_MAX))


#         self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN, self.df_lin[i][0]+h[1,0], self.DF_MAX))
#         # self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN, h[1,0]*self.fps, self.DF_DOT_MAX))



#         self.opti[i].subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[i][0]+self.a_lin[i][0]+h[0,0])*self.fps)
#         self.opti[i].subject_to((-self.u_prev[i][0]+self.a_lin[i][0]+h[0,0])*self.fps<=slack+self.A_DOT_MAX)

#         self.opti[i].subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[i][1]+self.df_lin[i][0]+h[1,0])*self.fps)
#         self.opti[i].subject_to((-self.u_prev[i][1]+self.df_lin[i][0]+h[1,0])*self.fps<=slack+self.DF_DOT_MAX)


#         if t_bar==0:

#             A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
#             B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
#             C_block=ca.MX((4+2*N_TV)*self.N, (2+2*N_TV)*self.N)
#             E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

#             A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][0][0] for k in range(N_TV)])
#             A_block[0:4, 4:4+2*N_TV]=B[0]@K[0]
#             B_block[0:4,0:2]=B[0]
#             C_block[0:4+2*N_TV,0:2+2*N_TV]=ca.diagcat(ca.vertcat(ca.MX.eye(2), ca.MX(2,2)), ca.MX.eye(2*N_TV))
#             E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E[0:4,0:4], *[self.Sigma_tv_sqrt[i][k][0][0] for k in range(N_TV)])
#             # E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
#             H=h[:,0]
#             c_ev=-K[0]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][0][0], self.y_tv_ref[i][k][0][0]) for k in range(N_TV)])
#             C=ca.vertcat(c_ev, *[c[k][0][0] for k in range(N_TV)])

#             for t in range(1,self.N):

#                 oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t], self.Q_tv[i][k][0][t-1]) for k in range(N_TV)]

#                 for k in range(N_TV):
#                     # Rot_TV

#                     soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
#                                              slack+2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
#                                                     @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
#                                                       +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:(2+2*N_TV)*t]@C)\
#                                                    +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))

#                     self.opti[i].subject_to(soc_constr>0)


#                 A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
#                 A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]


#                 B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
#                 B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


#                 C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
#                 C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(2+2*N_TV):(t+1)*(2+2*N_TV)]=ca.diagcat(ca.vertcat(ca.MX.eye(2), ca.MX(2,2)), ca.MX.eye(2*N_TV))
#                 c_ev=-K[t]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t]) for k in range(N_TV)])
#                 C=ca.vertcat(C, c_ev, *[c[k][0][t] for k in range(N_TV)])


#                 E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
#                 E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
#                 # E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][0][t] for k in range(N_TV)])
#                 E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E

#                 H=ca.vertcat(H, h[:,t])
#             nom_z_ev=ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H)
#             cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)

#             # pdb.set_trace()
#             cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),self.R))

#             nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
#             nom_df=H.reshape((2,self.N))[1,:]
#             nom_diff_df=ca.diff(nom_df+self.df_ref[i][:-1],1,1)
#             nom_da=H.reshape((2,self.N))[0,:]
#             nom_diff_a=ca.diff(nom_da+self.a_ref[i][:-1],1,1)

#             # pdb.set_trace()
#             self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
#                                                       nom_dv+self.v_lin[i][1:],
#                                                       self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


#             self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
#                                                       nom_df+self.df_lin[i][:-1],
#                                                       self.DF_MAX))


#             self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
#                                                       nom_da+self.a_lin[i][:-1],
#                                                       self.A_MAX))


#             self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN*self.DT-slack,
#                                                       nom_diff_a,
#                                                       self.A_DOT_MAX*self.DT+slack))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


#             self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN*self.DT-slack,
#                                                       nom_diff_df,
#                                                       self.DF_DOT_MAX*self.DT+slack))


#         elif t_bar<self.N-1:

#                 mode_map=list(product([*range(self.N_modes)],repeat=N_TV))
#                 mode_map=sorted([(sum([10**mode_map[i][j] for j in range(len(mode_map[i]))]),)+mode_map[i] for i in range(len(mode_map))])
#                 mode_map=[mode_map[i][1:] for i in range(len(mode_map))]
#                 if not self.noswitch_bl:
#                     seq=list(product([*range(self.N_modes**N_TV)],repeat=min(6,t_bar+1)))
#                     seq=seq[:min(self.N_seq_max, (self.N_modes**N_TV)**min(6,t_bar+1))]
#                     tail_seq=[[seq[j][-1]]*(self.N-min(6,t_bar+1)) for j in range(len(seq))]
#                     #                 pdb.set_trace()
#                     seq=[list(seq[i])+tail_seq[i] for i in range(len(seq))]

#                 else:
#                     seq=[self.N*[m] for m in range(self.N_modes**N_TV)]


#                 for s in range(len(seq)):


#                     A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
#                     B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
#                     C_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)
#                     E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)
#                     # pdb.set_trace()
#                     A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
#                     A_block[0:4, 4:4+2*N_TV]=B[0]@K[0]

#                     B_block[0:4,0:2]=B[0]
#                     C_block[:4+2*N_TV,0:4+2*N_TV]=ca.diagcat(ca.MX.eye(4), ca.MX.eye(2*N_TV))
#                     E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
#                     # E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E[0:4,0:4], *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
#                     H=h[:,0]
#                     c_ev=-B[0]@K[0]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][0]][k]][0], self.y_tv_ref[i][k][mode_map[seq[s][0]][k]][0]) for k in range(N_TV)])
#                     C=ca.vertcat(c_ev, *[c[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])

#                     for t in range(1,self.N):

#                         # oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]
#                         oa_ref=[self._oa_ev_ref([self.x_lin[i][t-1], self.x_lin[i][t]], [self.y_lin[i][t-1], self.y_lin[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]
#                         # oa_ref=[self._oa_ev_ref([x, x], [y, y], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]
#                         eval_oa_i.append([(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1] for k in range(N_TV)])

#                         for k in range(N_TV):

#                             soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
#                                                     2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
#                                                     @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
#                                                       +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:(4+2*N_TV)*t]@C)\
#                                                    +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]@(self.z_lin[i][:2, t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))#+self.z_lin[i][:2, t])))

#                             # soc_da=ca.soc(ca.horzcat(*[M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))], ))

#                             # pdb.set_trace()
#                             self.opti[i].subject_to(soc_constr>0)


#                         A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
#                         # A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
#                         # A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
#                         # c_ev=-B[t]@K[t]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])

#                         if t<t_bar or seq[s][t]==0:
#                             A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
#                             H=ca.vertcat(H, h[:,t])
#                             c_ev=-B[t]@K[t]@(ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])\
#                                     -C_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),:(4+2*N_TV)*t]@C)

#                             E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

#                             Gains=ca.horzcat(*[M[j] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))], K[t])
#                             noise=ca.diagcat(ca.kron(ca.MX.eye(t),E[0:4,0:4]), *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])
#                             soc_da=ca.soc(Gains[0,:]@noise, slack +0.5-H[-2])
#                             soc_df=ca.soc(Gains[1,:]@noise, slack+0.2-H[-1])



#                         else:
#                             A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t+seq[s][t]*(self.N-t_bar)]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
#                             H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])
#                             c_ev=-B[t]@K[t+seq[s][t]*(self.N-t_bar)]@(ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])\
#                                     -C_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),:(4+2*N_TV)*t]@C)
#                             E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))], ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

#                             Gains=ca.horzcat(*[M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))], K[t+seq[s][t]*(self.N-t_bar)])
#                             noise=ca.diagcat(ca.kron(ca.MX.eye(t),E[0:4,0:4]), *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])
#                             soc_da=ca.soc(Gains[0,:]@noise, slack+0.5-H[-2])
#                             soc_df=ca.soc(Gains[1,:]@noise, slack+0.2-H[-1])

#                         # self.opti[i].subject_to(soc_da>0)
#                         # self.opti[i].subject_to(soc_df>0)

#                         B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
#                         B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


#                         C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
#                         C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(ca.MX.eye(4), ca.MX.eye(2*N_TV))

#                         C=ca.vertcat(C, c_ev, *[c[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])

#                         E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
#                         # if t<t_bar or seq[s][t]==0:
#                         #     H=ca.vertcat(H, h[:,t])

#                         #     E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

#                         # else:
#                         #     H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])

#                         #     E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))], ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

#                         E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E
#                         # E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])

#                     nom_z_ev=ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H+C_block@C)
#                     nom_z_err=self.z_lin[i][:,1:].reshape((-1,1))-self.z_ref[i][:,1:].reshape((-1,1))+nom_z_ev
#                     nom_z_diff= ca.diff(nom_z_ev.reshape((4,-1)),1,1).reshape((-1,1))
#                     # cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)
#                     # cost_matrix_z=ca.kron(ca.diagcat(*[1**i for i in range(self.N)]),self.Q)
#                     cost_matrix_z=ca.diagcat(*[1**t*self.rot_costs[i][t] for t in range(self.N)])
#                     cost_matrix_u=ca.kron(ca.diagcat(*[1**i for i in range(self.N-1)]),self.R)

#                     nom_z_ev_i.append(nom_z_ev)
#                     nom_u_ev_i.append(H.reshape((2,self.N)))
#                     nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
#                     nom_df=H.reshape((2,self.N))[1,:]
#                     nom_diff_df=ca.diff(nom_df+self.df_lin[i][:-1],1,1)
#                     nom_da=H.reshape((2,self.N))[0,:]
#                     nom_diff_a=ca.diff(nom_da+self.a_lin[i][:-1],1,1)

#                     nom_diff_u=ca.diff(H.reshape((2,self.N)),1,1).reshape((-1,1))/self.DT

#                     cost+=RefTrajGenerator._quad_form(nom_z_ev, 10*cost_matrix_z)+\
#                           RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),ca.diag([0, 0])))+\
#                           RefTrajGenerator._quad_form(nom_z_diff,1000*cost_matrix_z[:(self.N-1)*4,:(self.N-1)*4])+\
#                           RefTrajGenerator._quad_form(nom_diff_u,10*cost_matrix_u)
#                           #+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),1*ca.MX.eye(2)))
#                     # pdb.set_trace()
#                     self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
#                                                               nom_dv+self.v_lin[i][1:],
#                                                               self.V_MAX+slack))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


#                     self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
#                                                               nom_df+self.df_lin[i][:-1],
#                                                               self.DF_MAX))


#                     self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
#                                                               nom_da+self.a_lin[i][:-1],
#                                                               self.A_MAX))


#                     self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN-slack,
#                                                               nom_diff_a/self.DT,
#                                                               self.A_DOT_MAX+slack))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


#                     self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN-slack,
#                                                               nom_diff_df/self.DT,
#                                                               self.DF_DOT_MAX+slack))



#         self.opti[i].minimize( cost )
#         self.nom_z_ev.append(nom_z_ev_i)
#         self.nom_u_ev.append(nom_u_ev_i)
#         self.eval_oa.append(eval_oa_i)

#     def solve(self, i):
#         st = time.time()
#         # pdb.set_trace()
#         try:

#             sol = self.opti[i].solve()

#             # Optimal solution.
#             u_control  = sol.value(self.policy[i][0][:,0])
#             v_tp1      = sol.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
#             is_feas     = True

#             z_lin_ev   = sol.value(self.z_lin[i])
#             u_lin_ev   = sol.value(self.u_lin[i])
#             z_ref_ev   = sol.value(self.z_ref[i])
#             # u_ref_ev   = sol.value(self.u_lin[i])
#             nom_z_ev   = [sol.value(x).reshape((4,-1))+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
#             nom_u_ev   = [sol.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

#             z_tv_ref    = np.array([sol.value(self.x_tv_ref[i][0][0]), sol.value(self.y_tv_ref[i][0][0])])

#             eval_oa     = np.array([sol.value(x[0]) for x in self.eval_oa[i]])


#         except:


#             # Suboptimal solution (e.g. timed out).

#             if self.opti[i].stats()['return_status']=='SUBOPTIMAL':
#                 u_control  = self.opti[i].debug.value(self.policy[i][0][:,0])
#                 v_tp1      = self.opti[i].debug.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
#                 is_feas     = True

#                 z_lin_ev   = self.opti[i].debug.value(self.z_lin[i])
#                 u_lin_ev   = self.opti[i].debug.value(self.u_lin[i])
#                 z_ref_ev   = self.opti[i].debug.value(self.z_ref[i])
#                 # u_ref_ev   = sol.value(self.u_lin[i])
#                 nom_z_ev   = [self.opti[i].debug.value(x).reshape((4,-1))+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
#                 nom_u_ev   = [self.opti[i].debug.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

#                 z_tv_ref    = np.array([self.opti[i].debug.value(self.x_tv_ref[i][0][0]), self.opti[i].debug.value(self.y_tv_ref[i][0][0])])
#                 # pdb.set_trace()
#                 eval_oa     = np.array([self.opti[i].debug.value(x[0]) for x in self.eval_oa[i]])
#             else:
#                 pdb.set_trace()
#                 if self.v_curr> 1:
#                     u_control  = np.array([self.a_brake-self.u_ref_val[0], -self.u_ref_val[1]])
#                     v_tp1      = self.v_curr+self.DT*self.a_brake
#                 else:
#                     u_control  = np.array([0., 0.])
#                     v_tp1      = self.v_next

#                 is_feas = False

#         solve_time = time.time() - st

#         sol_dict = {}
#         sol_dict['u_control']  = u_control  # control input to apply based on solution
#         sol_dict['v_next']     = v_tp1
#         sol_dict['optimal']    = is_feas
#              # whether the solution is optimal or not
#         if not is_feas:
#             sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
#         else:
#             sol_dict['solve_time'] = self.opti[i].stats()["t_wall_solver"]  # how long the solver took in seconds
#             sol_dict['nom_z_ev']= nom_z_ev
#             sol_dict['nom_u_ev']= nom_u_ev
#             sol_dict['z_lin']   = z_lin_ev
#             sol_dict['z_ref']   = z_ref_ev
#             sol_dict['z_tv_ref']= z_tv_ref
#             # pdb.set_trace()
#             if i!=0:
#              sol_dict['eval_oa'] = eval_oa[:self.N-1,:]

#         return sol_dict

#     def update(self, i, update_dict):
#         self._update_ev_initial_condition(i, *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
#         self._update_ev_rotated_costs(i, update_dict['Rs_ev'])
#         self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']] )
#         self._update_ev_reference(i, *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
#         self._update_ev_lin(i, *[update_dict[key] for key in ['x_lin', 'y_lin', 'psi_lin', 'v_lin', 'a_lin', 'df_lin']] )
#         self._update_tv_preds(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']], *[update_dict[key] for key in ['mus', 'sigmas']] )
#         self._update_previous_input(i, *[update_dict[key] for key in ['acc_prev', 'df_prev']] )
#         self._update_tv_shapes(i, update_dict['tv_shapes'])
#         self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
#         self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]
#         self.v_next=update_dict['v_ref'][1]
#         self.update_dict=update_dict
#         # pdb.set_trace()


#     def _update_ev_initial_condition(self, i, dx0, dy0, dpsi0, dvel0):
#         self.opti[i].set_value(self.dz_curr[i], ca.DM([dx0, dy0, dpsi0, dvel0]))

#     def _update_ev_rotated_costs(self, i, Rs_ev):
#         for t in range(self.N):
#             self.opti[i].set_value(self.rot_costs[i][t], ca.diagcat(Rs_ev[t].T@self.Q[:2,:2]@Rs_ev[t], self.Q[2:,2:]))



#     def _update_tv_shapes(self, i, Q_tv):
#         Q=self.Q_tv[i]
#         N_TV=1+int(i/self.t_bar_max)

#         for k in range(N_TV):
#             for j in range(self.N_modes):
#                 for t in range(self.N):
#                     if t==self.N-1:
#                         self.opti[i].set_value(Q[k][j][t], Q_tv[k][j][t-1])
#                     else:
#                         self.opti[i].set_value(Q[k][j][t], Q_tv[k][j][t])




#     def _update_tv_initial_condition(self, i, x_tv0, y_tv0):

#         N_TV=1+int(i/self.t_bar_max)
#         for k in range(N_TV):
#             self.opti[i].set_value(self.z_tv_curr[i][:,k], ca.DM([x_tv0[k], y_tv0[k]]))

#     def _update_ev_reference(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):
#         # pdb.set_trace()
#         self.opti[i].set_value(self.x_ref[i],   x_ref)
#         self.opti[i].set_value(self.y_ref[i],   y_ref)
#         self.opti[i].set_value(self.psi_ref[i], psi_ref)
#         self.opti[i].set_value(self.v_ref[i],   v_ref)
#         self.opti[i].set_value(self.a_ref[i],   a_ref)
#         self.opti[i].set_value(self.df_ref[i],   df_ref)

#     def _update_ev_lin(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):
#         # pdb.set_trace()
#         self.opti[i].set_value(self.x_lin[i],   x_ref)
#         self.opti[i].set_value(self.y_lin[i],   y_ref)
#         self.opti[i].set_value(self.psi_lin[i], psi_ref)
#         self.opti[i].set_value(self.v_lin[i],   v_ref)
#         self.opti[i].set_value(self.a_lin[i],   a_ref)
#         self.opti[i].set_value(self.df_lin[i],   df_ref)

#     def _update_tv_preds(self, i, x_tv0, y_tv0, mu_tv, sigma_tv):

#         N_TV=1+int(i/self.t_bar_max)
#         self._set_ATV_TV_dynamics(i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv)
#         self._set_TV_ref(i, N_TV, x_tv0, y_tv0, mu_tv)

#     def _update_previous_input(self, i, acc_prev, df_prev):
#         self.opti[i].set_value(self.u_prev[i], [acc_prev, df_prev])


class SMPC_MMPreds():

    def __init__(self,
                N            = 10,
                DT           = 0.2,
                L_F          = 1.7213,
                L_R          = 1.4987,
                V_MIN        = 0.0,
                V_MAX        = 15.0,
                A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
                A_MAX      =  2.0,
                DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
                DF_MAX     =  0.5,
                A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
                A_DOT_MAX  =  1.5,
                DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
                DF_DOT_MAX =  0.5,
                N_modes_MAX  =  3,
                N_TV_MAX     =  1,
                N_seq_MAX    =  100,
                T_BAR_MAX    =  6,
                TIGHTENING   =  1.64,
                NOISE_STD    =  [0.1, .1, .01, .1, 0.1], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                Q =[0.1*50., 0.005*500, 1*10., 0.1*10.], # weights on x, y, and v.
                R = [10., 1000],       # weights on inputs
                NS_BL_FLAG=False,
                fixed_risk=True,
                inv_cdf      = [np.array([[0.02, 1.35],[0.508, 0.91]]), np.array([[1.35,2.],[0.91, 0.978]])],
                fps = 20
                ):
        self.N=N
        self.DT=DT
        self.L_F=L_F
        self.L_R=L_R
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.A_MIN=A_MIN
        self.A_MAX=A_MAX
        self.DF_MIN=DF_MIN
        self.DF_MAX=DF_MAX
        self.A_DOT_MIN=A_DOT_MIN
        self.A_DOT_MAX=A_DOT_MAX
        self.DF_DOT_MIN=DF_DOT_MIN
        self.DF_DOT_MAX=DF_DOT_MAX
        self.N_modes=N_modes_MAX
        self.N_TV_max=N_TV_MAX
        self.N_seq_max=N_seq_MAX
        self.t_bar_max=T_BAR_MAX
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.a_brake=-7.0
        self.v_curr=0.0
        self.noswitch_bl=NS_BL_FLAG
        self.fps=fps

        self.fixed_risk=fixed_risk

        self.inv_cfd=[]

        for i in range(len(inv_cdf)):
            m=(inv_cdf[i][1,1]-inv_cdf[i][1,0])/(inv_cdf[i][0,1]-inv_cdf[i][0,0])
            c=inv_cdf[i][1,0]-m*inv_cdf[i][0,0]
            self.inv_cdfl.append([m,c])

        self.opti=[]

        self.z_ref=[]
        self.z_lin=[]
        self.u_prev=[]
        self.x_ref=[]
        self.y_ref=[]
        self.psi_ref=[]
        self.v_ref=[]
        self.u_ref=[]
        self.a_ref=[]
        self.df_ref=[]
        self.x_lin=[]
        self.y_lin=[]
        self.psi_lin=[]
        self.v_lin=[]
        self.u_lin=[]
        self.a_lin=[]
        self.df_lin=[]
        self.dz_curr=[]
        self.Sigma_tv_sqrt  =  []
        self.Q_tv = []

        self.T_tv=[]
        self.c_tv=[]

        self.x_tv_ref=[]
        self.y_tv_ref=[]
        self.z_tv_curr=[]
        self.rot_costs=[]

        self.policy=[]
        self.slacks=[]

        self.nom_z_ev = []
        self.nom_u_ev = []
        self.eval_oa=[]

        self.c_mmrstd=[]
        self.c_mmrprob=[]

        self.mmrisk_std= []
        self.mmrisk_prob=[]

        self.probs=[]



        p_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-3, 'PSDTol' : 1e-3}
        s_opts_grb = {'error_on_fail':0}


        for i in range((self.t_bar_max)*self.N_TV_max):
            self.opti.append(ca.Opti('conic'))
            self.opti[i].solver("gurobi", s_opts_grb, p_opts_grb)


            N_TV=1+int(i/self.t_bar_max)
            t_bar=i-(N_TV-1)*self.t_bar_max

            self.probs.append(self.opti[i].parameter(self.N_modes**N_TV))

            self.c_mmrstd.append(ca.DM([self.tight]*(self.N_modes**N_TV)))
            self.c_mmrprob.append(ca.DM([norm.cdf(self.tight)]*(self.N_modes**N_TV)))

            self.z_ref.append(self.opti[i].parameter(4, self.N+1))
            self.u_ref.append(self.opti[i].parameter(2, self.N+1))
            self.u_prev.append(self.opti[i].parameter(2))
            self.x_ref.append(self.z_ref[i][0, :])
            self.y_ref.append(self.z_ref[i][1, :] )
            self.psi_ref.append(self.z_ref[i][2, :] )
            self.v_ref.append(self.z_ref[i][3, :])

            self.a_ref.append(self.u_ref[i][0, :])
            self.df_ref.append(self.u_ref[i][1, :])

            self.z_lin.append(self.opti[i].parameter(4, self.N+1))
            self.u_lin.append(self.opti[i].parameter(2, self.N+1))
            self.x_lin.append(self.z_lin[i][0, :])
            self.y_lin.append(self.z_lin[i][1, :] )
            self.psi_lin.append(self.z_lin[i][2, :] )
            self.v_lin.append(self.z_lin[i][3, :])

            self.a_lin.append(self.u_lin[i][0, :])
            self.df_lin.append(self.u_lin[i][1, :])

            self.dz_curr.append(self.opti[i].parameter(4))
            self.slacks.append(self.opti[i].variable(1))

            if self.fixed_risk:
                self.mmrisk_std.append(self.opti[i].parameter(self.N_modes**N_TV))
                self.opti[i].set_value(self.mmrisk_std[i], self.c_mmrstd[i])
                self.mmrisk_prob.append(self.opti[i].parameter(self.N_modes**N_TV))
                self.opti[i].set_value(self.mmrisk_prob[i], self.c_mmrprob[i])
            else:
                self.mmrisk_std.append(self.opti[i].variable(self.N_modes**N_TV))
                self.mmrisk_prob.append(self.opti[i].variable(self.N_modes**N_TV))




            self.T_tv.append([[[self.opti[i].parameter(2,2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])
            self.c_tv.append([[[self.opti[i].parameter(2,1) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])



            self.x_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])
            self.y_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])

            self.Sigma_tv_sqrt.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])
            self.Q_tv.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])

            self.z_tv_curr.append(self.opti[i].parameter(2,N_TV))
            self.rot_costs.append([self.opti[i].parameter(4,4) for t in range(self.N)])
            self.policy.append(self._return_policy_class(i, N_TV, t_bar))
            self._add_constraints_and_cost(i, N_TV, t_bar)
            self.u_ref_val=np.zeros((2,1))
            self.v_next=np.array(5.)
            self._update_ev_initial_condition(i, 0., 0., np.pi*0., 5.0 )
            self._update_ev_rotated_costs(i, self.N*[np.identity(2)])
            self._update_tv_initial_condition(i, N_TV*[20.0], N_TV*[20.0] )
            self._update_ev_reference(i, [self.DT *5.0* (x) for x in range(self.N+1)],
                                      [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
            self._update_ev_lin(i, [self.DT *5.0* (x) for x in range(self.N+1)],
                                      [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
            self._update_tv_preds(i, N_TV*[20.0], N_TV*[20.0], N_TV*[20*np.ones((self.N_modes, self.N, 2))], N_TV*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
            self._update_previous_input(i, 0.0, 0.0)
            self._update_tv_shapes(i, N_TV*[self.N_modes*[self.N*[0.1*np.identity(2)]]])
            sol=self.solve(i)




    def _return_policy_class(self, i, N_TV, t_bar):

        if t_bar == 0 or t_bar==self.N-1:
            M=[[[self.opti[i].variable(2, 4)] for n in range(t)] for t in range(self.N)]
            K=[[[self.opti[i].variable(2,2) for k in range(N_TV)]] for t in range(1,self.N)]
            h=[[self.opti[i].variable(2, 1)] for t in range(self.N)]

            M_stack=[ca.vertcat(*[ca.horzcat(*[M[t][j][0] for j in range(t)], ca.DM(2,4*(self.N-t))) for t in range(self.N)])]
            h_stack=[ca.vertcat(*[h[t][0] for t in range(self.N)])]
            K_stack=[[ca.diagcat(ca.DM(2,2),*[K[t][0][k] for t in range(self.N-1)]) for k in range(N_TV)]]

        else:
            h=[[self.opti[i].variable(2,1) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(self.N)]
            M=[[[self.opti[i].variable(2, 4) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for j in range(t)] for t in range(self.N)]
            K=[[[self.opti[i].variable(2,2) for k in range(N_TV)] for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(1,self.N)]

            M_stack=[ca.vertcat(*[ca.horzcat(*[M[t][j][m*int(t>=t_bar)] for j in range(t)], ca.DM(2,4*(self.N-t))) for t in range(self.N)]) for m in range(self.N_modes)]
            h_stack=[ca.vertcat(*[h[t][m*int(t>=t_bar)] for t in range(self.N)]) for m in range(self.N_modes)]
            K_stack=[[ca.diagcat(ca.DM(2,2),*[K[t-1][m*int(t>=t_bar)][k] for t in range(1,self.N)]) for k in range(N_TV)] for m in range(self.N_modes)]


        return h_stack,M_stack,K_stack

    def _get_ATV_TV_dynamics(self, i, N_TV):

        ## Fit time-varying model for target vehicle using GMM parameters


        T=self.T_tv[i]
        c=self.c_tv[i]

        T_block=[[ca.MX((self.N+1)*2,2) for j in range(self.N_modes)] for k in range(N_TV)]
        C_block=[[ca.MX((self.N+1)*2,1) for j in range(self.N_modes)] for k in range(N_TV)]
        F_block=[[ca.MX((self.N+1)*2,self.N*2) for j in range(self.N_modes)] for k in range(N_TV)]

        F=ca.DM.eye(2)*self.noise_std[-1]


        for k in range(N_TV):
            for j in range(self.N_modes):
                T_block[k][j][0:2,:]=ca.DM.eye(2)
                for t in range(self.N):
                    T_block[k][j][(t+1)*2:(t+2)*2,:]=T[k][j][t]@T_block[k][j][t*2:(t+1)*2,:]
                    C_block[k][j][(t+1)*2:(t+2)*2,:]=T[k][j][t]@C_block[k][j][t*2:(t+1)*2,:]+c[k][j][t]
                    F_block[k][j][(t+1)*2:(t+2)*2,:]=T[k][j][t]@F_block[k][j][t*2:(t+1)*2,:]
                    if t==0:
                        F_block[k][j][(t+1)*2:(t+2)*2,t*2:(t+1)*2]=self.Sigma_tv_sqrt[i][k][j][t]
                    else:
                        F_block[k][j][(t+1)*2:(t+2)*2,t*2:(t+1)*2]=F

        return T_block, C_block, F_block


    def _set_TV_ref(self, i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv):


        T=self.T_tv[i]
        c=self.c_tv[i]

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N+1):
                    if t==0:
                        self.opti[i].set_value(self.x_tv_ref[i][k][j][0], x_tv0[k])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][0], y_tv0[k])
                        self.opti[i].set_value(T[k][j][t], np.identity(2))
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-np.hstack((x_tv0[k],y_tv0[k])))
                        e_val,e_vec= np.linalg.eigh(0.1*sigma_tv[k][j,t,:,:])
                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], e_vec@np.diag(np.sqrt(e_val))@e_vec.T)
                    else:
                        self.opti[i].set_value(self.x_tv_ref[i][k][j][t], mu_tv[k][j,t-1,0])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][t], mu_tv[k][j,t-1,1])
                        if t<self.N:
                            e_val,e_vec= np.linalg.eigh(0.1*sigma_tv[k][j,t,:,:])
                            e_valp,e_vecp= np.linalg.eigh(0.1*sigma_tv[k][j,t-1,:,:])
                            Ttv=e_vec@np.diag(np.sqrt(e_val))@e_vec.T@e_vecp@np.diag(np.sqrt(e_valp)**(-1))@e_vecp.T

                            self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], e_vec@np.diag(np.sqrt(e_val))@e_vec.T )
                            self.opti[i].set_value(T[k][j][t], Ttv)
                            self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-Ttv@mu_tv[k][j, t-1, :])





    def _get_LTV_EV_dynamics(self, i, N_TV):

        A=[ca.MX.eye(4) for n in range(self.N+1)]
        B=[ca.MX(np.zeros((4, 2))) for n in range(self.N+1)]

        A_block=ca.MX(4*(self.N+1),4)
        B_block=ca.MX(4*(self.N+1),2*self.N)
        E_block=ca.MX(4*(self.N+1),4*self.N)

        A_block[0:4,0:4]=ca.DM.eye(4)

        E=ca.MX(4, 4)
        E[0:4,0:4]=(ca.DM.eye(4))@ca.diag(self.noise_std[0:4])
        # E[4:, 4:]=ca.DM.eye(2*N_TV)*self.noise_std[-1]


        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_lin[i][t])**2



            A[t][0,2]+=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_lin[i][t]+beta))
            A[t][1,2]+=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_lin[i][t]+beta))
            A[t][2,3]+=self.DT*(1./self.L_R*ca.sin(beta))
            # A[t]=ca.expm(A[t])


            B[t][0,1]=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta)*dbeta)
            B[t][1,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta)*dbeta)
            B[t][2,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0

            A_block[(t+1)*4:(t+2)*4, :]=A[t]
            B_block[(t+1)*4:(t+2)*4,:]=A[t]@B_block[t*4:(t+1)*4,:]
            B_block[(t+1)*4:(t+2)*4,t*2:(t+1)*2]=B[t]
            E_block[(t+1)*4:(t+2)*4,:]=A[t]@E_block[t*4:(t+1)*4,:]
            E_block[(t+1)*4:(t+2)*4,t*4:(t+1)*4]=E



        return A_block,B_block,E_block

    def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv, Q):

         ## EV reference chosen to be AM of consecutive positions in provided reference. Heuristically helps to align hyperplanes along path
            x_ev_avg=0.5*(x_ev[0]+x_ev[1])
            y_ev_avg=0.5*(y_ev[0]+y_ev[1])
            # x_ev_avg=x_ev[1]
            # y_ev_avg=y_ev[1]


            x_ref_ev=x_tv+(x_ev_avg-x_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
            y_ref_ev=y_tv+(y_ev_avg-y_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))

            return ca.vertcat(x_ref_ev,y_ref_ev)




    def _add_constraints_and_cost(self, i, N_TV, t_bar):


        nom_z_ev_i = []
        nom_u_ev_i = []
        eval_oa_i=[]

        mmr_std=self.mmrisk_std[i]
        mmr_p=self.mmrisk_prob[i]



        [A_block,B_block,E_block]=self._get_LTV_EV_dynamics(i, N_TV)
        [T_block,C_block,F_block]=self._get_ATV_TV_dynamics(i, N_TV)

        [h,M,K]=self.policy[i]

        slack=self.slacks[i]
        cost = 10*slack@slack
        x=0.5*(self.dz_curr[i][0]+self.x_lin[i][0]+self.x_lin[i][-1])
        y=0.5*(self.dz_curr[i][0]+self.y_lin[i][0]+self.y_lin[i][-1])
        self.opti[i].subject_to(slack>=0)


        self.opti[i].subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[i][0]+self.a_lin[i][0]+h[0][0,0])*self.fps)
        self.opti[i].subject_to((-self.u_prev[i][0]+self.a_lin[i][0]+h[0][0,0])*self.fps<=slack+self.A_DOT_MAX)

        self.opti[i].subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[i][1]+self.df_lin[i][0]+h[0][1,0])*self.fps)
        self.opti[i].subject_to((-self.u_prev[i][1]+self.df_lin[i][0]+h[0][1,0])*self.fps<=slack+self.DF_DOT_MAX)

        total_prob=0
        for j in range(1+(-1+self.N_modes)*(t_bar>0)):
            if not self.fixed_risk:
                self.opti[i].subject_to(self.opti[i].bounded(0.0000001, mmr_std[j],3.0))
                total_prob+=mmr_p[j]*self.probs[i][j]
                for m,c in self.inv_cdfl:
                    self.opti.subject_to(mmr_p[j]<=m*mmr_std[j]+c)
                    self.opti.subject_to(self.opti.bounded(0.5, mmr_p[j],1.))


            for t in range(1,self.N):

                # oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]
                oa_ref=[self._oa_ev_ref([self.x_lin[i][t-1], self.x_lin[i][t]], [self.y_lin[i][t-1], self.y_lin[i][t]], self.x_tv_ref[i][k][j][t], self.y_tv_ref[i][k][j][t], self.Q_tv[i][k][j][t-1]) for k in range(N_TV)]
                # oa_ref=[self._oa_ev_ref([x, x], [y, y], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]

                eval_oa_i.append([(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][j][t], self.y_tv_ref[i][k][j][t])).T@self.Q_tv[i][k][j][t-1] for k in range(N_TV)])

                for k in range(N_TV):

                    z=-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][j][t], self.y_tv_ref[i][k][j][t])).T@self.Q_tv[i][k][j][t-1]@(ca.horzcat(B_block[t*4:t*4+2,:]@M[j]+mmr_std[j]*E_block[t*4:t*4+2,:], *[B_block[t*4:t*4+2,:]@K[j][l]@F_block[l][j][:2*self.N,:]-int(l==k)*mmr_std[j]*F_block[k][j][t*2:(t+1)*2,:] for l in range(N_TV)]))

                    y=+2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][j][t], self.y_tv_ref[i][k][j][t])).T@self.Q_tv[i][k][j][t-1]@(A_block[t*4:t*4+2,:]@self.dz_curr[i]+B_block[t*4:t*4+2,:]@h[j]-T_block[k][j][t*2:(t+1)*2,:]@self.z_tv_curr[i][:,k]-C_block[k][j][t*2:(t+1)*2,:])\
                                                   +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][j][t], self.y_tv_ref[i][k][j][t])).T@self.Q_tv[i][k][j][t-1]@(self.z_lin[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][j][t], self.y_tv_ref[i][k][j][t]))

                    soc_constr=ca.soc(z,y)

                    self.opti[i].subject_to(soc_constr>0)


            nom_z_ev=A_block@self.dz_curr[i]+B_block@h[j]
            nom_z_err=nom_z_ev[4:,:]+self.z_lin[i][:,1:].reshape((-1,1))-self.z_ref[i][:,1:].reshape((-1,1))
            nom_z_diff= ca.diff(nom_z_ev.reshape((4,-1)),1,1).reshape((-1,1))


            cost_matrix_z=ca.diagcat(self.Q, *[1**t*self.rot_costs[i][t] for t in range(self.N)])
            cost_matrix_u=ca.kron(ca.diagcat(*[1**i for i in range(self.N-1)]),self.R)

            nom_z_ev_i.append(nom_z_ev)
            nom_u_ev_i.append(h[j].reshape((2,self.N)))
            nom_dv=nom_z_ev.reshape((4,-1))[3,:]
            nom_df=h[j].reshape((2,self.N))[1,:]
            nom_diff_df=ca.diff(nom_df+self.df_lin[i][:-1],1,1)
            nom_da=h[j].reshape((2,self.N))[0,:]
            nom_diff_a=ca.diff(nom_da+self.a_lin[i][:-1],1,1)

            nom_diff_u=ca.diff(h[j].reshape((2,self.N)),1,1).reshape((-1,1))/self.DT

            cost+=RefTrajGenerator._quad_form(nom_z_ev, 10*cost_matrix_z)+\
                  RefTrajGenerator._quad_form(h[j],ca.kron(ca.DM.eye(self.N),ca.diag([0, 0])))+\
                  RefTrajGenerator._quad_form(nom_z_diff,100*cost_matrix_z[4:,4:])+\
                  RefTrajGenerator._quad_form(nom_diff_u,10*cost_matrix_u)
                  #+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),1*ca.MX.eye(2)))

            self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                      nom_dv[1:]+self.v_lin[i][1:],
                                                      self.V_MAX+slack))


            self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                                                      nom_df+self.df_lin[i][:-1],
                                                      self.DF_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                                                      nom_da+self.a_lin[i][:-1],
                                                      self.A_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN-slack,
                                                      nom_diff_a/self.DT,
                                                      self.A_DOT_MAX+slack))


            self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN-slack,
                                                      nom_diff_df/self.DT,
                                                      self.DF_DOT_MAX+slack))
        if not self.fixed_risk:
            self.opti[i].subject_to(total_prob>=self.c_mmrprob[i][0])

        self.opti[i].minimize( cost )
        self.nom_z_ev.append(nom_z_ev_i)
        self.nom_u_ev.append(nom_u_ev_i)
        self.eval_oa.append(eval_oa_i)

    def solve(self, i):
        st = time.time()

        try:
            # pdb.set_trace()
            sol = self.opti[i].solve()

            # Optimal solution.
            u_control  = sol.value(self.policy[i][0][0][:2,0])
            v_tp1      = sol.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0][0,0])
            is_feas     = True

            z_lin_ev   = sol.value(self.z_lin[i])
            u_lin_ev   = sol.value(self.u_lin[i])
            z_ref_ev   = sol.value(self.z_ref[i])

            nom_z_ev   = [sol.value(x).reshape((4,-1))[:,1:]+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
            nom_u_ev   = [sol.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

            z_tv_ref    = np.array([sol.value(self.x_tv_ref[i][0][0]), sol.value(self.y_tv_ref[i][0][0])])
            eval_oa     = np.array([sol.value(x[0]) for x in self.eval_oa[i]])


        except:


            # Suboptimal solution (e.g. timed out).

            if self.opti[i].stats()['return_status']=='SUBOPTIMAL':
                u_control  = self.opti[i].debug.value(self.policy[i][0][0][:2,0])
                v_tp1      = self.opti[i].debug.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0][0,0])
                is_feas     = True

                z_lin_ev   = self.opti[i].debug.value(self.z_lin[i])
                u_lin_ev   = self.opti[i].debug.value(self.u_lin[i])
                z_ref_ev   = self.opti[i].debug.value(self.z_ref[i])

                nom_z_ev   = [self.opti[i].debug.value(x).reshape((4,-1))[:,1:]+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
                nom_u_ev   = [self.opti[i].debug.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

                z_tv_ref    = np.array([self.opti[i].debug.value(self.x_tv_ref[i][0][0]), self.opti[i].debug.value(self.y_tv_ref[i][0][0])])
                eval_oa     = np.array([self.opti[i].debug.value(x[0]) for x in self.eval_oa[i]])


            else:
                # pdb.set_trace()
                if self.v_curr> 1:
                    u_control  = np.array([self.a_brake-self.u_ref_val[0], -self.u_ref_val[1]])
                    v_tp1      = self.v_curr+self.DT*self.a_brake
                else:
                    u_control  = np.array([0., 0.])
                    v_tp1      = self.v_next

                is_feas = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_feas
             # whether the solution is optimal or not
        if not is_feas:
            sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
        else:
            sol_dict['solve_time'] = self.opti[i].stats()["t_wall_solver"]  # how long the solver took in seconds
            sol_dict['nom_z_ev']= nom_z_ev
            sol_dict['nom_u_ev']= nom_u_ev
            sol_dict['z_lin']   = z_lin_ev
            sol_dict['z_ref']   = z_ref_ev
            sol_dict['z_tv_ref']= z_tv_ref

            if i!=0:
             sol_dict['eval_oa'] = eval_oa[:self.N-1,:]

        return sol_dict

    def update(self, i, update_dict):
        self._update_ev_initial_condition(i, *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_ev_rotated_costs(i, update_dict['Rs_ev'])
        self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']] )
        self._update_ev_reference(i, *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_ev_lin(i, *[update_dict[key] for key in ['x_lin', 'y_lin', 'psi_lin', 'v_lin', 'a_lin', 'df_lin']] )
        self._update_tv_preds(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']], *[update_dict[key] for key in ['mus', 'sigmas']] )
        self._update_previous_input(i, *[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self._update_tv_shapes(i, update_dict['tv_shapes'])
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
        self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]
        self.v_next=update_dict['v_ref'][1]
        self.update_dict=update_dict

        N_TV=1+int(i/self.t_bar_max)
        if "probs" in update_dict.keys():
            self.opti[i].set_value(self.probs[i], update_dict["probs"])
        else:
            self.opti[i].set_value(self.probs[i], np.ones(self.N_modes**N_TV)/(self.N_modes**N_TV))



    def _update_ev_initial_condition(self, i, dx0, dy0, dpsi0, dvel0):
        self.opti[i].set_value(self.dz_curr[i], ca.DM([dx0, dy0, dpsi0, dvel0]))

    def _update_ev_rotated_costs(self, i, Rs_ev):
        for t in range(self.N):
            self.opti[i].set_value(self.rot_costs[i][t], ca.diagcat(Rs_ev[t]@self.Q[:2,:2]@Rs_ev[t].T, self.Q[2:,2:]))



    def _update_tv_shapes(self, i, Q_tv):
        Q=self.Q_tv[i]
        N_TV=1+int(i/self.t_bar_max)

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N):
                    if t==self.N-1:
                        self.opti[i].set_value(Q[k][j][t], Q_tv[k][j][t-1])
                    else:
                        self.opti[i].set_value(Q[k][j][t], Q_tv[k][j][t])




    def _update_tv_initial_condition(self, i, x_tv0, y_tv0):

        N_TV=1+int(i/self.t_bar_max)
        for k in range(N_TV):
            self.opti[i].set_value(self.z_tv_curr[i][:,k], ca.DM([x_tv0[k], y_tv0[k]]))

    def _update_ev_reference(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):

        self.opti[i].set_value(self.x_ref[i],   x_ref)
        self.opti[i].set_value(self.y_ref[i],   y_ref)
        self.opti[i].set_value(self.psi_ref[i], psi_ref)
        self.opti[i].set_value(self.v_ref[i],   v_ref)
        self.opti[i].set_value(self.a_ref[i],   a_ref)
        self.opti[i].set_value(self.df_ref[i],   df_ref)

    def _update_ev_lin(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):

        self.opti[i].set_value(self.x_lin[i],   x_ref)
        self.opti[i].set_value(self.y_lin[i],   y_ref)
        self.opti[i].set_value(self.psi_lin[i], psi_ref)
        self.opti[i].set_value(self.v_lin[i],   v_ref)
        self.opti[i].set_value(self.a_lin[i],   a_ref)
        self.opti[i].set_value(self.df_lin[i],   df_ref)

    def _update_tv_preds(self, i, x_tv0, y_tv0, mu_tv, sigma_tv):

        N_TV=1+int(i/self.t_bar_max)
        self._set_TV_ref(i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv)

    def _update_previous_input(self, i, acc_prev, df_prev):
        self.opti[i].set_value(self.u_prev[i], [acc_prev, df_prev])

class SMPC_MMPreds_OBCA():

    def __init__(self,
                N            = 10,
                DT           = 0.2,
                L_F          = 1.7213,
                L_R          = 1.4987,
                V_MIN        = 0.0,
                V_MAX        = 15.0,
                A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
                A_MAX      =  2.0,
                DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
                DF_MAX     =  0.5,
                A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
                A_DOT_MAX  =  1.5,
                DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
                DF_DOT_MAX =  0.5,
                N_modes_MAX  =  3,
                N_TV_MAX     =  1,
                N_seq_MAX    =  100,
                T_BAR_MAX    =  6,
                TIGHTENING   =  1.5,
                NOISE_STD    =  [0.1, .1, .01, .1, .5], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                Q =[0.1*50., 0.01*50, 1*10., 0.1*10.], # weights on x, y, and v.
                R = [10., 1000],       # weights on inputs
                NS_BL_FLAG=False,
                fps = 20
                ):
        self.N=N
        self.DT=DT
        self.L_F=L_F
        self.L_R=L_R
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.A_MIN=A_MIN
        self.A_MAX=A_MAX
        self.DF_MIN=DF_MIN
        self.DF_MAX=DF_MAX
        self.A_DOT_MIN=A_DOT_MIN
        self.A_DOT_MAX=A_DOT_MAX
        self.DF_DOT_MIN=DF_DOT_MIN
        self.DF_DOT_MAX=DF_DOT_MAX
        self.N_modes=N_modes_MAX
        self.N_TV_max=N_TV_MAX
        self.N_seq_max=N_seq_MAX
        self.t_bar_max=T_BAR_MAX
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.a_brake=-7.
        self.v_curr=0.0
        self.noswitch_bl=NS_BL_FLAG
        self.fps=fps

        self.G=ca.DM([[1., 0.],[-1.,0. ], [0, 1.],[0.,-1. ]])
        self.g=ca.DM([[2.5],[2.5], [1.4],[1.4]])


        self.opti=[]

        self.z_ref=[]
        self.z_lin=[]
        self.u_prev=[]
        self.x_ref=[]
        self.y_ref=[]
        self.psi_ref=[]
        self.v_ref=[]
        self.u_ref=[]
        self.a_ref=[]
        self.df_ref=[]
        self.x_lin=[]
        self.y_lin=[]
        self.psi_lin=[]
        self.v_lin=[]
        self.u_lin=[]
        self.a_lin=[]
        self.df_lin=[]
        self.dz_curr=[]
        self.Sigma_tv_sqrt  =  []
        self.R_tv = []
        self.R_ev=[]

        self.T_tv=[]
        self.c_tv=[]

        self.x_tv_ref=[]
        self.y_tv_ref=[]
        self.z_tv_curr=[]
        self.rot_costs=[]

        self.M_prev=[]
        self.K_prev=[]
        self.h_prev=[]
        self.lmbd_prev=[]
        self.nu_prev=[]

        self.lmbd_dual_var=[]
        self.nu_dual_var=[]

        self.policy=[]
        self.slacks=[]

        self.nom_z_ev = []
        self.nom_u_ev = []



        p_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-3, 'PSDTol' : 1e-3}
        s_opts_grb = {'error_on_fail':0}


        for i in range((self.t_bar_max)*self.N_TV_max):
            self.opti.append(ca.Opti('conic'))
            self.opti[i].solver("gurobi", s_opts_grb, p_opts_grb)


            N_TV=1+int(i/self.t_bar_max)
            t_bar=i-(N_TV-1)*self.t_bar_max

            self.z_ref.append(self.opti[i].parameter(4, self.N+1))
            self.u_ref.append(self.opti[i].parameter(2, self.N+1))
            self.u_prev.append(self.opti[i].parameter(2))
            self.x_ref.append(self.z_ref[i][0, :])
            self.y_ref.append(self.z_ref[i][1, :] )
            self.psi_ref.append(self.z_ref[i][2, :] )
            self.v_ref.append(self.z_ref[i][3, :])

            self.a_ref.append(self.u_ref[i][0, :])
            self.df_ref.append(self.u_ref[i][1, :])

            self.z_lin.append(self.opti[i].parameter(4, self.N+1))
            self.u_lin.append(self.opti[i].parameter(2, self.N+1))
            self.x_lin.append(self.z_lin[i][0, :])
            self.y_lin.append(self.z_lin[i][1, :] )
            self.psi_lin.append(self.z_lin[i][2, :] )
            self.v_lin.append(self.z_lin[i][3, :])

            self.a_lin.append(self.u_lin[i][0, :])
            self.df_lin.append(self.u_lin[i][1, :])

            self.dz_curr.append(self.opti[i].parameter(4))
            self.slacks.append(self.opti[i].variable(1))

            if t_bar == 0 or t_bar==self.N-1:
                self.M_prev.append([self.opti[i].parameter(2, 4) for j in range(int((self.N-1)*self.N/2))])
                self.K_prev.append([self.opti[i].parameter(2,2*N_TV) for j in range(self.N)])
                self.h_prev.append(self.opti[i].parameter(2, self.N))

            else:
                self.M_prev.append([self.opti[i].parameter(2, 4) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))])
                self.K_prev.append([self.opti[i].parameter(2,2*N_TV) for j in range(t_bar+(self.N_modes*N_TV)*(self.N-t_bar))])
                self.h_prev.append(self.opti[i].parameter(2, t_bar+(self.N_modes*N_TV)*(self.N-t_bar)))


            self.lmbd_prev.append([[self.opti[i].parameter(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])
            self.nu_prev.append([[self.opti[i].parameter(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])


            self.lmbd_dual_var.append([[self.opti[i].variable(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])
            self.nu_dual_var.append([[self.opti[i].variable(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])

            self.T_tv.append([[[self.opti[i].parameter(2,2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])
            self.c_tv.append([[[self.opti[i].parameter(2,1) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])



            self.x_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])
            self.y_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])

            self.Sigma_tv_sqrt.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])
            self.R_tv.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])
            self.R_ev.append([ self.opti[i].parameter(2, 2) for _ in range(self.N) ])

            self.z_tv_curr.append(self.opti[i].parameter(2,N_TV))
            self.rot_costs.append([self.opti[i].parameter(4,4) for t in range(self.N)])
            self.policy.append(self._return_policy_class(i, N_TV, t_bar))
            self._add_constraints_and_cost(i, N_TV, t_bar)
            self.u_ref_val=np.zeros((2,1))
            self.v_next=np.array(5.)
            # self._update_ev_initial_condition(i, 0., 0., np.pi*0., 5.0 )
            # self._update_ev_rotated_costs(i, self.N*[np.identity(2)])
            # self._update_tv_initial_condition(i, N_TV*[20.0], N_TV*[20.0] )
            # self._update_ev_reference(i, [self.DT *5.0* (x) for x in range(self.N+1)],
            #                           [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
            # self._update_ev_lin(i, [self.DT *5.0* (x) for x in range(self.N+1)],
            #                           [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
            # self._update_tv_preds(i, N_TV*[20.0], N_TV*[20.0], N_TV*[20*np.ones((self.N_modes, self.N, 2))], N_TV*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
            # self._update_previous_input(i, 0.0, 0.0)
            # self._update_tv_shapes(i, N_TV*[self.N_modes*[self.N*[0.1*np.identity(2)]]])
            # sol=self.solve(i)




    def _return_policy_class(self, i, N_TV, t_bar):

        if t_bar == 0 or t_bar==self.N-1:
            M=[[[self.opti[i].variable(2, 4)] for n in range(t)] for t in range(self.N)]
            K=[[[self.opti[i].variable(2,2) for k in range(N_TV)]] for t in range(self.N)]
            h=[[self.opti[i].variable(2, 1)] for t in range(self.N)]

            self.M_prev.append([self.opti[i].parameter(2*self.N, 4*self.N)])
            self.K_prev.append([[self.opti[i].parameter(2*self.N,2*self.N) for k in range(N_TV)]])
            self.h_prev.append([[self.opti[i].parameter(2*self.N, 1)]])


            M_stack=[ca.vertcat(*[ca.horzcat(*[M[t][j][0] for j in range(t)], ca.DM(2,4*(self.N-t))) for t in range(self.N)])]
            h_stack=[ca.vertcat(*[h[t][0] for t in range(self.N)])]
            K_stack=[[ca.diagcat(*[K[t][0][k] for t in range(self.N)]) for k in range(N_TV)]]

        else:
            h=[[self.opti[i].variable(2,1) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(self.N)]
            M=[[[self.opti[i].variable(2, 4) for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for j in range(t)] for t in range(self.N)]
            K=[[[self.opti[i].variable(2,2) for k in range(N_TV)] for n in range(1+(-1+self.N_modes)*int(t>=t_bar))] for t in range(self.N)]

            self.M_prev.append([self.opti[i].parameter(2*self.N, 4*self.N) for m in range(self.N_modes)])
            self.K_prev.append([[self.opti[i].parameter(2*self.N,2*self.N) for k in range(N_TV)] for m in range(self.N_modes) ])
            self.h_prev.append([[self.opti[i].parameter(2*self.N, 1)] for m in range(self.N_modes)])

            M_stack=[ca.vertcat(*[ca.horzcat(*[M[t][j][m*int(t>=t_bar)] for j in range(t)], ca.DM(2,4*(self.N-t))) for t in range(self.N)]) for m in range(self.N_modes)]
            h_stack=[ca.vertcat(*[h[t][m*int(t>=t_bar)] for t in range(self.N)]) for m in range(self.N_modes)]
            K_stack=[[ca.diagcat(*[K[t][m*int(t>=t_bar)][k] for t in range(self.N)]) for k in range(N_TV)] for m in range(self.N_modes)]


        return h_stack,M_stack,K_stack

    def _set_ATV_TV_dynamics(self, i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv):

        ## Fit time-varying model for target vehicle using GMM parameters


        T=self.T_tv[i]
        c=self.c_tv[i]

        self.T_block=[[ca.MX((self.N+1)*2,2) for j in range(self.N_modes)] for k in range(N_TV)]
        self.C_block=[[ca.MX((self.N+1)*2,1) for j in range(self.N_modes)] for k in range(N_TV)]
        self.F_block=[[ca.MX((self.N+1)*2,self.N*2) for j in range(self.N_modes)] for k in range(N_TV)]

        F=ca.DM.eye(2*N_TV)*self.noise_std[-1]


        for k in range(N_TV):
            for j in range(self.N_modes):
                self.T_block[k][j][0:2,:]=ca.DM.eye(2)
                for t in range(self.N):
                    if t==0:
                        self.opti[i].set_value(T[k][j][t], np.identity(2))
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-np.hstack((x_tv0[k],y_tv0[k])))
                        e_val,e_vec= np.linalg.eigh(sigma_tv[k][j,t,:,:])
                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], e_vec@np.diag(np.sqrt(e_val))@e_vec.T)

                    else:

                        e_val,e_vec= np.linalg.eigh(sigma_tv[k][j,t,:,:])
                        e_valp,e_vecp= np.linalg.eigh(sigma_tv[k][j,t-1,:,:])
                        Ttv=e_vec@np.diag(np.sqrt(e_val))@e_vec.T@e_vecp@np.diag(np.sqrt(e_valp)**(-1))@e_vecp.T

                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], ca.chol(sigma_tv[k][j,t,:,:]) )
                        self.opti[i].set_value(T[k][j][t], Ttv)
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-Ttv@mu_tv[k][j, t-1, :])

                    self.T_block[k][j][(t+1)*2:(t+2)*2,:]=T[k][j][t]@self.T_block[k][j][t*2:(t+1)*2,:]
                    self.C_block[k][j][(t+1)*2:(t+2)*2,:]=T[k][j][t]@self.C_block[k][j][t*2:(t+1)*2,:]+c[k][j][t]
                    self.F_block[k][j][(t+1)*2:(t+2)*2,:]=T[k][j][t]@self.F_block[k][j][t*2:(t+1)*2,:]
                    self.F_block[k][j][(t+1)*2:(t+2)*2,t*2:(t+1)*2]=F



    def _set_TV_ref(self, i, N_TV, x_tv0, y_tv0, mu_tv):

        # Mean target positions to be used for linearizing obstacle avoidance constraints

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N+1):
                    if t==0:
                        self.opti[i].set_value(self.x_tv_ref[i][k][j][0], x_tv0[k])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][0], y_tv0[k])
                    else:
                        self.opti[i].set_value(self.x_tv_ref[i][k][j][t], mu_tv[k][j,t-1,0])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][t], mu_tv[k][j,t-1,1])




    def _get_LTV_EV_dynamics(self, i, N_TV):

        A=[ca.MX.eye(4) for n in range(self.N+1)]
        B=[ca.MX(np.zeros((4, 2))) for n in range(self.N+1)]

        A_block=ca.MX(4*(self.N+1),4)
        B_block=ca.MX(4*(self.N+1),2*self.N)
        E_block=ca.MX(4*(self.N+1),4*self.N)

        A_block[0:4,0:4]=ca.DM.eye(4)

        E=ca.MX(4, 4)
        E[0:4,0:4]=(ca.DM.eye(4))@ca.diag(self.noise_std[0:4])
        # E[4:, 4:]=ca.DM.eye(2*N_TV)*self.noise_std[-1]


        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_lin[i][t])**2



            A[t][0,2]+=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_lin[i][t]+beta))
            A[t][1,2]+=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_lin[i][t]+beta))
            A[t][2,3]+=self.DT*(1./self.L_R*ca.sin(beta))
            # A[t]=ca.expm(A[t])


            B[t][0,1]=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta)*dbeta)
            B[t][1,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta)*dbeta)
            B[t][2,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0

            A_block[(t+1)*4:(t+2)*4, :]=A[t]
            B_block[(t+1)*4:(t+2)*4,:]=A[t]@B_block[t*4:(t+1)*4,:]
            B_block[(t+1)*4:(t+2)*4,t*2:(t+1)*2]=B[t]
            E_block[(t+1)*4:(t+2)*4,:]=A[t]@E_block[t*4:(t+1)*4,:]
            E_block[(t+1)*4:(t+2)*4,t*4:(t+1)*4]=E



        return A_block,B_block,E_block

    def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv, Q):

         ## EV reference chosen to be AM of consecutive positions in provided reference. Heuristically helps to align hyperplanes along path
            x_ev_avg=0.5*(x_ev[0]+x_ev[1])
            y_ev_avg=0.5*(y_ev[0]+y_ev[1])
            # x_ev_avg=x_ev[1]
            # y_ev_avg=y_ev[1]


            x_ref_ev=x_tv+(x_ev_avg-x_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
            y_ref_ev=y_tv+(y_ev_avg-y_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))

            return ca.vertcat(x_ref_ev,y_ref_ev)




    def _add_constraints_and_cost(self, i, N_TV, t_bar):


        nom_z_ev_i = []
        nom_u_ev_i = []


        T=self.T_tv[i]
        c=self.c_tv[i]
        [A_block,B_block,E_block]=self._get_LTV_EV_dynamics(i, N_TV)
        T_block=self.T_block
        C_block=self.C_block
        F_block=self.F_block

        [h,M,K]=self.policy[i]

        h_prev=self.h_prev[i]
        K_prev=self.K_prev[i]
        M_prev=self.M_prev[i]

        slack=self.slacks[i]

        cost = 10*slack@slack
        x=0.5*(self.dz_curr[i][0]+self.x_lin[i][0]+self.x_lin[i][-1])
        y=0.5*(self.dz_curr[i][0]+self.y_lin[i][0]+self.y_lin[i][-1])
        self.opti[i].subject_to(slack>=0)


        self.opti[i].subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[i][0]+self.a_lin[i][0]+h[0][0,0])*self.fps)
        self.opti[i].subject_to((-self.u_prev[i][0]+self.a_lin[i][0]+h[0][0,0])*self.fps<=slack+self.A_DOT_MAX)

        self.opti[i].subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[i][1]+self.df_lin[i][0]+h[0][1,0])*self.fps)
        self.opti[i].subject_to((-self.u_prev[i][1]+self.df_lin[i][0]+h[0][1,0])*self.fps<=slack+self.DF_DOT_MAX)


        if t_bar==0:

            self.Q_tv=self.R_tv

            for t in range(1,self.N):

                R_ev=self.R_ev[i][t-1].T

                for k in range(N_TV):

                    lmbd=self.lmbd_dual_var[i][k][0][:,t-1].T
                    nu=self.nu_dual_var[i][k][0][:,t-1].T
                    lmbd_prev=self.lmbd_prev[i][k][0][:,t-1].T
                    nu_prev=self.nu_prev[i][k][0][:,t-1].T
                    R_tv=self.R_tv[i][k][0][t-1].T
                    # pdb.set_trace()
                    z=(lmbd_prev@self.G@R_ev)@(ca.horzcat(B_block[t*4:t*4+2,:]@M[0]+E_block[t*4:t*4+2,:], *[B_block[t*4:t*4+2,:]@K[0][l]@F_block[l][0]-int(l==k)*F_block[k][0][t*2:(t+1)*2,:] for l in range(N_TV)]))\
                      +((lmbd-lmbd_prev)@self.G@R_ev)@(ca.horzcat(B_block[t*4:t*4+2,:]@M_prev[0]+E_block[t*4:t*4+2,:], *[B_block[t*4:t*4+2,:]@K_prev[0][l]@F_block[l][0]-int(l==k)*F_block[k][0][t*2:(t+1)*2,:] for l in range(N_TV)]))

                    y=(lmbd_prev@self.G@R_ev)@(A_block[t*4:t*4+2,:]@self.dz_curr[i]+B_block[t*4:t*4+2,:]@h[0]-T_block[k][0][t*2:(t+1)*2,:]@self.z_tv_curr[i][:,k]-C_block[k][0][t*2:(t+1)*2,:])\
                      +((lmbd-lmbd_prev)@self.G@R_ev)@(A_block[t*4:t*4+2,:]@self.dz_curr[i]+B_block[t*4:t*4+2,:]@h_prev[0]-T_block[k][0][t*2:(t+1)*2,:]@self.z_tv_curr[i][:,k]-C_block[k][0][t*2:(t+1)*2,:])\
                      -(lmbd+nu)@self.g-lmbd@self.G@R_ev@self.z_lin[i][:2, t]-.01

                    soc_constr=ca.soc(self.tight*z,y)


                    self.opti[i].subject_to(soc_constr>0)
                    self.opti[i].subject_to(lmbd>=0)
                    self.opti[i].subject_to(nu>=0)
                    self.opti[i].subject_to(lmbd@self.G@R_ev+nu@self.G@R_tv==0)
                    self.opti[i].subject_to(ca.norm_2(lmbd@self.G@R_ev)<=1)

            nom_z_ev=A_block@self.dz_curr[i]+B_block@h[0]
            cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)

            cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(h[0],ca.kron(ca.DM.eye(self.N),self.R))

            nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
            nom_df=h[0].reshape((2,self.N))[1,:]
            nom_diff_df=ca.diff(nom_df+self.df_ref[i][:-1],1,1)
            nom_da=h[0].reshape((2,self.N))[0,:]
            nom_diff_a=ca.diff(nom_da+self.a_ref[i][:-1],1,1)

            self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                      nom_dv+self.v_lin[i][1:],
                                                      self.V_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                                                      nom_df+self.df_lin[i][:-1],
                                                      self.DF_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                                                      nom_da+self.a_lin[i][:-1],
                                                      self.A_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN*self.DT-slack,
                                                      nom_diff_a,
                                                      self.A_DOT_MAX*self.DT+slack))

            self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN*self.DT-slack,
                                                      nom_diff_df,
                                                      self.DF_DOT_MAX*self.DT+slack))


        elif t_bar<self.N-1:

                for j in range(self.N_modes):

                    for t in range(1,self.N+1):

                        R_ev=self.R_ev[i][t-1].T

                        for k in range(N_TV):

                            lmbd=self.lmbd_dual_var[i][k][j][:,t-1].T
                            nu=self.nu_dual_var[i][k][j][:,t-1].T
                            lmbd_prev=self.lmbd_prev[i][k][j][:,t-1].T
                            nu_prev=self.nu_prev[i][k][j][:,t-1].T
                            R_tv=self.R_tv[i][k][j][t-1].T
                            # pdb.set_trace()
                            z=(lmbd_prev@self.G@R_ev)@(ca.horzcat(B_block[t*4:t*4+2,:]@M[j]+E_block[t*4:t*4+2,:], *[B_block[t*4:t*4+2,:]@K[j][l]@F_block[l][j]-int(l==k)*F_block[k][j][t*2:(t+1)*2,:] for l in range(N_TV)]))\
                              +((lmbd-lmbd_prev)@self.G@R_ev)@(ca.horzcat(B_block[t*4:t*4+2,:]@M_prev[j]+E_block[t*4:t*4+2,:], *[B_block[t*4:t*4+2,:]@K_prev[j][l]@F_block[l][j]-int(l==k)*F_block[k][j][t*2:(t+1)*2,:] for l in range(N_TV)]))

                            y=(lmbd_prev@self.G@R_ev)@(A_block[t*4:t*4+2,:]@self.dz_curr[i]+B_block[t*4:t*4+2,:]@h[j]-T_block[k][j][t*2:(t+1)*2,:]@self.z_tv_curr[i][:,k]-C_block[k][j][t*2:(t+1)*2,:])\
                              +((lmbd-lmbd_prev)@self.G@R_ev)@(A_block[t*4:t*4+2,:]@self.dz_curr[i]+B_block[t*4:t*4+2,:]@h_prev[j]-T_block[k][j][t*2:(t+1)*2,:]@self.z_tv_curr[i][:,k]-C_block[k][j][t*2:(t+1)*2,:])\
                              -(lmbd+nu)@self.g-lmbd@self.G@R_ev@self.z_lin[i][:2, t]-.01



                            soc_constr=ca.soc(self.tight*z,y)


                            self.opti[i].subject_to(soc_constr>0)
                            self.opti[i].subject_to(lmbd>=0)
                            self.opti[i].subject_to(nu>=0)
                            self.opti[i].subject_to(lmbd@self.G@R_ev+nu@self.G@R_tv==0)
                            self.opti[i].subject_to(ca.norm_2(lmbd@self.G@R_ev)<=1)


                    nom_z_ev=A_block@self.dz_curr[i]+B_block@h[j]
                    nom_z_err=self.z_lin[i][:,1:].reshape((-1,1))-self.z_ref[i][:,1:].reshape((-1,1))+nom_z_ev
                    nom_z_diff= ca.diff(nom_z_ev.reshape((4,-1)),1,1).reshape((-1,1))


                    cost_matrix_z=ca.diagcat(*[1**t*self.rot_costs[i][t] for t in range(self.N)])
                    cost_matrix_u=ca.kron(ca.diagcat(*[1**i for i in range(self.N-1)]),self.R)

                    nom_z_ev_i.append(nom_z_ev)
                    nom_u_ev_i.append(h[j].reshape((2,self.N)))
                    nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
                    nom_df=h[j].reshape((2,self.N))[1,:]
                    nom_diff_df=ca.diff(nom_df+self.df_lin[i][:-1],1,1)
                    nom_da=h[j].reshape((2,self.N))[0,:]
                    nom_diff_a=ca.diff(nom_da+self.a_lin[i][:-1],1,1)

                    nom_diff_u=ca.diff(h[j].reshape((2,self.N)),1,1).reshape((-1,1))/self.DT

                    cost+=RefTrajGenerator._quad_form(nom_z_err, 10*cost_matrix_z)+\
                          RefTrajGenerator._quad_form(h[j],ca.kron(ca.DM.eye(self.N),ca.diag([0., 0.])))+\
                          RefTrajGenerator._quad_form(nom_z_diff,1000*cost_matrix_z[:(self.N-1)*4,:(self.N-1)*4])+\
                          RefTrajGenerator._quad_form(nom_diff_u,10*cost_matrix_u)
                          #+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),1*ca.MX.eye(2)))

                    self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                              nom_dv+self.v_lin[i][1:],
                                                              self.V_MAX+slack))


                    self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                                                              nom_df+self.df_lin[i][:-1],
                                                              self.DF_MAX))


                    self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                                                              nom_da+self.a_lin[i][:-1],
                                                              self.A_MAX))


                    self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN-slack,
                                                              nom_diff_a/self.DT,
                                                              self.A_DOT_MAX+slack))


                    self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN-slack,
                                                              nom_diff_df/self.DT,
                                                              self.DF_DOT_MAX+slack))



        self.opti[i].minimize( cost )
        self.nom_z_ev.append(nom_z_ev_i)
        self.nom_u_ev.append(nom_u_ev_i)
        # self.eval_oa.append(eval_oa_i)

    def solve(self, i):
        st = time.time()

        N_TV=1+int(i/self.t_bar_max)
        t_bar=i-(N_TV-1)*self.t_bar_max

        try:
            # pdb.set_trace()
            sol = self.opti[i].solve()

            # Optimal solution.
            u_control  = sol.value(self.policy[i][0][0][:2,0])
            v_tp1      = sol.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0][0,0])
            is_feas     = True

            z_lin_ev   = sol.value(self.z_lin[i])
            u_lin_ev   = sol.value(self.u_lin[i])
            z_ref_ev   = sol.value(self.z_ref[i])



            if t_bar == 0 or t_bar==self.N-1:
                M_opt=[sol.value(self.policy[i][1][0])]
                K_opt=[[sol.value(self.policy[i][2][0][k]) for k in range(N_TV)]]
                h_opt=[sol.value(self.policy[i][0][0])]

            else:
                M_opt=[sol.value(self.policy[i][1][j]) for j in range(self.N_modes)]
                K_opt=[[sol.value(self.policy[i][2][j][k]) for k in range(N_TV)] for j in range(self.N_modes)]
                h_opt=[sol.value(self.policy[i][0][j]) for j in range(self.N_modes)]

            lmbd_opt    = [[sol.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]
            nu_opt     = [[sol.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]

            nom_z_ev   = [sol.value(x).reshape((4,-1))+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
            nom_u_ev   = [sol.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

            z_tv_ref    = np.array([sol.value(self.x_tv_ref[i][0][0]), sol.value(self.y_tv_ref[i][0][0])])



        except:


            # Suboptimal solution (e.g. timed out).

            if self.opti[i].stats()['return_status']=='SUBOPTIMAL':
                u_control  = self.opti[i].debug.value(self.policy[i][0][:,0])
                v_tp1      = self.opti[i].debug.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
                is_feas     = True

                z_lin_ev   = self.opti[i].debug.value(self.z_lin[i])
                u_lin_ev   = self.opti[i].debug.value(self.u_lin[i])
                z_ref_ev   = self.opti[i].debug.value(self.z_ref[i])

                nom_z_ev   = [self.opti[i].debug.value(x).reshape((4,-1))+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
                nom_u_ev   = [self.opti[i].debug.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

                if t_bar == 0 or t_bar==self.N-1:
                    M_opt=[self.opti[i].debug.value(self.policy[i][1][0])]
                    K_opt=[[self.opti[i].debug.value(self.policy[i][2][0][k]) for k in range(N_TV)]]
                    h_opt=[self.opti[i].debug.value(self.policy[i][0][0])]

                else:
                    M_opt=[self.opti[i].debug.value(self.policy[i][1][j]) for j in range(self.N_modes)]
                    K_opt=[[self.opti[i].debug.value(self.policy[i][2][j][k]) for k in range(N_TV)] for j in range(self.N_modes)]
                    h_opt=[self.opti[i].debug.value(self.policy[i][0][j]) for j in range(self.N_modes)]

                lmbd_opt    = [[self.opti[i].debug.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]
                nu_opt     = [[self.opti[i].debug.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]
                z_tv_ref    = np.array([self.opti[i].debug.value(self.x_tv_ref[i][0][0]), self.opti[i].debug.value(self.y_tv_ref[i][0][0])])


            else:
                # pdb.set_trace()
                if self.v_curr> 1:
                    u_control  = np.array([self.a_brake-self.u_ref_val[0], -self.u_ref_val[1]])
                    v_tp1      = self.v_curr+self.DT*self.a_brake
                else:
                    u_control  = np.array([0., 0.])
                    v_tp1      = self.v_next

                is_feas = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_feas
             # whether the solution is optimal or not
        if not is_feas:
            sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
        else:
            sol_dict['solve_time'] = self.opti[i].stats()["t_wall_solver"]  # how long the solver took in seconds
            sol_dict['nom_z_ev']= nom_z_ev
            sol_dict['nom_u_ev']= nom_u_ev
            sol_dict['h_opt']=h_opt
            sol_dict['M_opt']=M_opt
            sol_dict['K_opt']=K_opt
            sol_dict['lmbd_opt']=lmbd_opt
            sol_dict['nu_opt']=nu_opt
            sol_dict['z_lin']   = z_lin_ev
            sol_dict['z_ref']   = z_ref_ev
            sol_dict['z_tv_ref']= z_tv_ref



        return sol_dict

    def update(self, i, update_dict):
        self._update_ev_initial_condition(i, *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_ev_rotated_costs(i, update_dict['Rs_ev'])
        self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']] )
        self._update_ev_reference(i, *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_ev_lin(i, *[update_dict[key] for key in ['x_lin', 'y_lin', 'psi_lin', 'v_lin', 'a_lin', 'df_lin']] )
        self._update_tv_preds(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']], *[update_dict[key] for key in ['mus', 'sigmas']] )
        self._update_previous_input(i, *[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self._update_tv_shapes(i, update_dict['tv_shapes'])
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
        self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]
        self.v_next=update_dict['v_ref'][1]
        self.update_dict=update_dict

        


        N_TV=1+int(i/self.t_bar_max)
        t_bar=i-(N_TV-1)*self.t_bar_max

        if t_bar == 0 or t_bar==self.N-1:

            self.opti[i].set_value(self.M_prev[i][0], np.zeros((2*self.N,4*self.N)))
            self.opti[i].set_value(self.h_prev[i][0], np.zeros((2*self.N,1)))
            if 'ws' in update_dict.keys():
                self.opti[i].set_value(self.M_prev[i][0], update_dict['ws'][1][0])
                self.opti[i].set_value(self.h_prev[i][0], update_dict['ws'][0][0])
                # self.opti[i].set_initial(self.policy[i][2][j], update_dict['ws'][2][j])
            for k in range(N_TV):
                self.opti[i].set_value(self.K_prev[i][0][k], np.zeros((2*self.N,2*self.N)))
                if 'ws' in update_dict.keys():
                    self.opti[i].set_value(self.K_prev[i][0][k], update_dict['ws'][2][0][k])
                    # self.opti[i].set_initial(self.policy[i][1][j], update_dict['ws'][1][j])

                # self.opti[i].set_initial(self.policy[i][0], update_dict['ws'][0])
        else:
            for j in range(self.N_modes):
                self.opti[i].set_value(self.M_prev[i][j], np.zeros((2*self.N,4*self.N)))
                self.opti[i].set_value(self.h_prev[i][j], np.zeros((2*self.N,1)))
                if 'ws' in update_dict.keys():
                    self.opti[i].set_value(self.M_prev[i][j], update_dict['ws'][1][j])
                    self.opti[i].set_value(self.h_prev[i][j], update_dict['ws'][0][j])
                    # self.opti[i].set_initial(self.policy[i][2][j], update_dict['ws'][2][j])
                    # self.opti[i].set_initial(self.policy[i][0], update_dict['ws'][0])

                for k in range(N_TV):
                    self.opti[i].set_value(self.K_prev[i][j][k], np.zeros((2*self.N,2*self.N)))
                    if 'ws' in update_dict.keys():
                        self.opti[i].set_value(self.K_prev[i][j][k], update_dict['ws'][2][j][k])
                        # self.opti[i].set_initial(self.policy[i][1][j], update_dict['ws'][1][j])




        for j in range(self.N_modes):
            for k in range(N_TV):
                # self.opti[i].set_initial(self.lmbd_dual_var[i][k][j],  0.5*ca.DM.ones(4,self.N))
                # self.opti[i].set_initial(self.nu_dual_var[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_value(self.lmbd_prev[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_value(self.nu_prev[i][k][j],  0.5*ca.DM.ones(4,self.N))
                if 'ws' in update_dict.keys():
                    # self.opti[i].set_initial(self.lmbd_dual_var[i][k][j], update_dict['ws'][3][k][j])
                    # self.opti[i].set_initial(self.nu_dual_var[i][k][j], update_dict['ws'][4][k][j])

                    self.opti[i].set_value(self.lmbd_prev[i][k][j], update_dict['ws'][3][k][j])
                    self.opti[i].set_value(self.nu_prev[i][k][j], update_dict['ws'][4][k][j])




    def _update_ev_initial_condition(self, i, dx0, dy0, dpsi0, dvel0):
        self.opti[i].set_value(self.dz_curr[i], ca.DM([dx0, dy0, dpsi0, dvel0]))

    def _update_ev_rotated_costs(self, i, Rs_ev):
        for t in range(self.N):
            self.opti[i].set_value(self.rot_costs[i][t], ca.diagcat(Rs_ev[t].T@self.Q[:2,:2]@Rs_ev[t], self.Q[2:,2:]))
            self.opti[i].set_value(self.R_ev[i][t], Rs_ev[t])



    def _update_tv_shapes(self, i, Q_tv):
        R=self.R_tv[i]
        N_TV=1+int(i/self.t_bar_max)

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N):
                    if t==self.N-1:
                        self.opti[i].set_value(R[k][j][t], Q_tv[k][j][t-1])
                    else:
                        self.opti[i].set_value(R[k][j][t], Q_tv[k][j][t])




    def _update_tv_initial_condition(self, i, x_tv0, y_tv0):

        N_TV=1+int(i/self.t_bar_max)
        for k in range(N_TV):
            self.opti[i].set_value(self.z_tv_curr[i][:,k], ca.DM([x_tv0[k], y_tv0[k]]))

    def _update_ev_reference(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):

        self.opti[i].set_value(self.x_ref[i],   x_ref)
        self.opti[i].set_value(self.y_ref[i],   y_ref)
        self.opti[i].set_value(self.psi_ref[i], psi_ref)
        self.opti[i].set_value(self.v_ref[i],   v_ref)
        self.opti[i].set_value(self.a_ref[i],   a_ref)
        self.opti[i].set_value(self.df_ref[i],   df_ref)

    def _update_ev_lin(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):

        self.opti[i].set_value(self.x_lin[i],   x_ref)
        self.opti[i].set_value(self.y_lin[i],   y_ref)
        self.opti[i].set_value(self.psi_lin[i], psi_ref)
        self.opti[i].set_value(self.v_lin[i],   v_ref)
        self.opti[i].set_value(self.a_lin[i],   a_ref)
        self.opti[i].set_value(self.df_lin[i],   df_ref)

    def _update_tv_preds(self, i, x_tv0, y_tv0, mu_tv, sigma_tv):

        N_TV=1+int(i/self.t_bar_max)
        self._set_ATV_TV_dynamics(i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv)
        self._set_TV_ref(i, N_TV, x_tv0, y_tv0, mu_tv)

    def _update_previous_input(self, i, acc_prev, df_prev):
        self.opti[i].set_value(self.u_prev[i], [acc_prev, df_prev])




# class SMPC_MMPreds_OBCA():

    def __init__(self,
                N            = 10,
                DT           = 0.2,
                L_F          = 1.7213,
                L_R          = 1.4987,
                V_MIN        = 0.0,
                V_MAX        = 15.0,
                A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
                A_MAX      =  2.5,
                DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
                DF_MAX     =  0.5,
                A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
                A_DOT_MAX  =  1.5,
                DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
                DF_DOT_MAX =  0.5,
                N_modes_MAX  =  1,
                N_TV_MAX     =  1,
                N_seq_MAX    =  100,
                T_BAR_MAX    =  6,
                EPS          =  0.05,
                TIGHTENING   =  1.64,
                NOISE_STD    =  [0.1, .1, .01, .1, .1], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                Q =[0.1*50., 0.001*50, 1*10., 0.1*10.], # weights on x, y, and v.
                R = [10., 1000],       # weights on inputs
                NS_BL_FLAG=False,
                pol_mode=2,
                fps = 20
                ):
        self.N=N
        self.DT=DT
        self.L_F=L_F
        self.L_R=L_R
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.A_MIN=A_MIN
        self.A_MAX=A_MAX
        self.DF_MIN=DF_MIN
        self.DF_MAX=DF_MAX
        self.A_DOT_MIN=A_DOT_MIN
        self.A_DOT_MAX=A_DOT_MAX
        self.DF_DOT_MIN=DF_DOT_MIN
        self.DF_DOT_MAX=DF_DOT_MAX
        self.N_modes=N_modes_MAX
        self.N_TV_max=N_TV_MAX
        self.N_seq_max=N_seq_MAX
        self.t_bar_max=T_BAR_MAX
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.a_brake=-6.
        self.v_curr=0.0
        self.noswitch_bl=NS_BL_FLAG
        self.fps=fps
        self.pol_mode=pol_mode

        self.G=ca.DM([[1., 0.],[-1.,0. ], [0, 1.],[0.,-1. ]])
        self.g=ca.DM([[2.45],[2.45], [1.35],[1.35]])

        if self.pol_mode!=3:
            self.tight=stats.norm.ppf(1-EPS)
        else:
            self.tight=((1-EPS)/EPS)**0.5

        self.opti=[]

        self.z_ref=[]
        self.z_lin=[]
        self.u_prev=[]
        self.x_ref=[]
        self.y_ref=[]
        self.psi_ref=[]
        self.v_ref=[]
        self.u_ref=[]
        self.a_ref=[]
        self.df_ref=[]
        self.x_lin=[]
        self.y_lin=[]
        self.psi_lin=[]
        self.v_lin=[]
        self.u_lin=[]
        self.a_lin=[]
        self.df_lin=[]
        self.dz_curr=[]
        self.Sigma_tv_sqrt  =  []
        self.R_tv = []
        self.R_ev=[]

        self.T_tv=[]
        self.c_tv=[]

        self.x_tv_ref=[]
        self.y_tv_ref=[]
        self.z_tv_curr=[]
        self.rot_costs=[]

        self.M_prev=[]
        self.K_prev=[]
        self.h_prev=[]
        self.lmbd_prev=[]
        self.nu_prev=[]

        self.t_1norm=[]

        self.lmbd_dual_var=[]
        self.nu_dual_var=[]

        self.policy=[]
        self.slacks=[]

        self.nom_z_ev = []
        self.nom_u_ev = []



        p_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-3, 'PSDTol' : 1e-3}
        s_opts_grb = {'error_on_fail':0}


        for i in range((self.t_bar_max)*self.N_TV_max):
            self.opti.append(ca.Opti('conic'))
            self.opti[i].solver("gurobi", s_opts_grb, p_opts_grb)


            N_TV=1+int(i/self.t_bar_max)
            t_bar=i-(N_TV-1)*self.t_bar_max

            self.z_ref.append(self.opti[i].parameter(4, self.N+1))
            self.u_ref.append(self.opti[i].parameter(2, self.N+1))
            self.u_prev.append(self.opti[i].parameter(2))
            self.x_ref.append(self.z_ref[i][0, :])
            self.y_ref.append(self.z_ref[i][1, :] )
            self.psi_ref.append(self.z_ref[i][2, :] )
            self.v_ref.append(self.z_ref[i][3, :])

            self.a_ref.append(self.u_ref[i][0, :])
            self.df_ref.append(self.u_ref[i][1, :])

            self.z_lin.append(self.opti[i].parameter(4, self.N+1))
            self.u_lin.append(self.opti[i].parameter(2, self.N+1))
            self.x_lin.append(self.z_lin[i][0, :])
            self.y_lin.append(self.z_lin[i][1, :] )
            self.psi_lin.append(self.z_lin[i][2, :] )
            self.v_lin.append(self.z_lin[i][3, :])

            self.a_lin.append(self.u_lin[i][0, :])
            self.df_lin.append(self.u_lin[i][1, :])

            self.dz_curr.append(self.opti[i].parameter(4))
            self.slacks.append(self.opti[i].variable(1))
            if self.pol_mode==1:
                self.t_1norm.append([[self.opti[i].variable(self.N*(4+N_TV*2),self.N) for j in range(self.N_modes)] for k in range(N_TV)])

            if t_bar == 0 or t_bar==self.N-1:
                self.M_prev.append([self.opti[i].parameter(2, 4) for j in range(int((self.N-1)*self.N/2))])
                self.K_prev.append([self.opti[i].parameter(2,2*N_TV) for j in range(self.N)])
                self.h_prev.append(self.opti[i].parameter(2, self.N))

            else:
                self.M_prev.append([self.opti[i].parameter(2, 4) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))])
                self.K_prev.append([self.opti[i].parameter(2,2*N_TV) for j in range(t_bar+(self.N_modes*N_TV)*(self.N-t_bar))])
                self.h_prev.append(self.opti[i].parameter(2, t_bar+(self.N_modes*N_TV)*(self.N-t_bar)))


            self.lmbd_prev.append([[self.opti[i].parameter(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])
            self.nu_prev.append([[self.opti[i].parameter(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])


            self.lmbd_dual_var.append([[self.opti[i].variable(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])
            self.nu_dual_var.append([[self.opti[i].variable(4,self.N) for j in range(self.N_modes)] for k in range(N_TV)])

            self.T_tv.append([[[self.opti[i].parameter(2,2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])
            self.c_tv.append([[[self.opti[i].parameter(2,1) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])



            self.x_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])
            self.y_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])

            self.Sigma_tv_sqrt.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])
            self.R_tv.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])
            self.R_ev.append([ self.opti[i].parameter(2, 2) for _ in range(self.N) ])

            self.z_tv_curr.append(self.opti[i].parameter(2,N_TV))
            self.rot_costs.append([self.opti[i].parameter(4,4) for t in range(self.N)])
            self.policy.append(self._return_policy_class(i, N_TV, t_bar))
            self._add_constraints_and_cost(i, N_TV, t_bar)
            self.u_ref_val=np.zeros((2,1))
            self.v_next=np.array(5.)
            # self._update_ev_initial_condition(i, 0., 0., np.pi*0., 5.0 )
            # self._update_ev_rotated_costs(i, self.N*[np.identity(2)])
            # self._update_tv_initial_condition(i, N_TV*[20.0], N_TV*[20.0] )
            # self._update_ev_reference(i, [self.DT *5.0* (x) for x in range(self.N+1)],
            #                           [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
            # self._update_ev_lin(i, [self.DT *5.0* (x) for x in range(self.N+1)],
            #                           [self.DT *0.0* (x) for x in range(self.N+1)],(self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
            # self._update_tv_preds(i, N_TV*[20.0], N_TV*[20.0], N_TV*[20*np.ones((self.N_modes, self.N, 2))], N_TV*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
            # self._update_previous_input(i, 0.0, 0.0)
            # self._update_tv_shapes(i, N_TV*[self.N_modes*[self.N*[0.1*np.identity(2)]]])
            # sol=self.solve(i)




    def _return_policy_class(self, i, N_TV, t_bar):

        if t_bar == 0 or t_bar==self.N-1:
            M=[self.opti[i].variable(2, 4) for j in range(int((self.N-1)*self.N/2))]
            K=[self.opti[i].variable(2,2*N_TV) for j in range(self.N)]
            h=self.opti[i].variable(2, self.N)

        else:
            M=[self.opti[i].variable(2, 4) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
            # M=[ca.DM(2, 4) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]

            K=[self.opti[i].variable(2,2*N_TV) for j in range(t_bar+(self.N_modes*N_TV)*(self.N-t_bar))]
            h=self.opti[i].variable(2, t_bar+(self.N_modes*N_TV)*(self.N-t_bar))

        return h,K,M

    def _set_ATV_TV_dynamics(self, i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv):

        ## Fit time-varying model for target vehicle using GMM parameters


        T=self.T_tv[i]
        c=self.c_tv[i]

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N):
                    if t==0:
                        self.opti[i].set_value(T[k][j][t], np.identity(2))
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-np.hstack((x_tv0[k],y_tv0[k])))
                        e_val,e_vec= np.linalg.eigh(sigma_tv[k][j,t,:,:])
                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], e_vec@np.diag(np.sqrt(e_val))@e_vec.T )
                    else:

                        e_val,e_vec= np.linalg.eigh(sigma_tv[k][j,t,:,:])
                        e_valp,e_vecp= np.linalg.eigh(sigma_tv[k][j,t-1,:,:])
                        Ttv=e_vec@np.diag(np.sqrt(e_val))@e_vec.T@e_vecp@np.diag(np.sqrt(e_valp)**(-1))@e_vecp.T

                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], ca.chol(sigma_tv[k][j,t,:,:]) )
                        self.opti[i].set_value(T[k][j][t], Ttv)
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-Ttv@mu_tv[k][j, t-1, :])



    def _set_TV_ref(self, i, N_TV, x_tv0, y_tv0, mu_tv):

        # Mean target positions to be used for linearizing obstacle avoidance constraints

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N+1):
                    if t==0:
                        self.opti[i].set_value(self.x_tv_ref[i][k][j][0], x_tv0[k])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][0], y_tv0[k])
                    else:
                        self.opti[i].set_value(self.x_tv_ref[i][k][j][t], mu_tv[k][j,t-1,0])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][t], mu_tv[k][j,t-1,1])




    def _get_LTV_EV_dynamics(self, i, N_TV):

        A=[ca.MX.eye(4) for n in range(self.N+1)]
        B=[ca.MX(np.zeros((4, 2))) for n in range(self.N+1)]

        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[i][t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_lin[i][t])**2



            A[t][0,2]+=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_lin[i][t]+beta))
            A[t][1,2]+=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_lin[i][t]+beta))
            A[t][2,3]+=self.DT*(1./self.L_R*ca.sin(beta))
            # A[t]=ca.expm(A[t])


            B[t][0,1]=self.DT*(-ca.fmax(self.v_lin[i][t],0.001)*ca.sin(self.psi_lin[i][t]+beta)*dbeta)
            B[t][1,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)*ca.cos(self.psi_lin[i][t]+beta)*dbeta)
            B[t][2,1]=self.DT*(ca.fmax(self.v_lin[i][t],0.001)/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0


        E=ca.MX(4+2*N_TV, 4+2*N_TV)
        E[0:4,0:4]=(ca.DM.eye(4))@ca.diag(self.noise_std[0:4])
        E[4:, 4:]=ca.DM.eye(2*N_TV)*self.noise_std[-1]

        return A,B,E

    def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv, Q):

         ## EV reference chosen to be AM of consecutive positions in provided reference. Heuristically helps to align hyperplanes along path
            x_ev_avg=0.5*(x_ev[0]+x_ev[1])
            y_ev_avg=0.5*(y_ev[0]+y_ev[1])
            # x_ev_avg=x_ev[1]
            # y_ev_avg=y_ev[1]


            x_ref_ev=x_tv+(x_ev_avg-x_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
            y_ref_ev=y_tv+(y_ev_avg-y_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))

            return ca.vertcat(x_ref_ev,y_ref_ev)




    def _add_constraints_and_cost(self, i, N_TV, t_bar):


        nom_z_ev_i = []
        nom_u_ev_i = []


        T=self.T_tv[i]
        c=self.c_tv[i]
        [A,B,E]=self._get_LTV_EV_dynamics(i, N_TV)
        [h,K,M]=self.policy[i]
        h_prev=self.h_prev[i]
        K_prev=self.K_prev[i]
        M_prev=self.M_prev[i]
        slack=self.slacks[i]
        cost = 10*slack@slack
        x=0.5*(self.dz_curr[i][0]+self.x_lin[i][0]+self.x_lin[i][-1])
        y=0.5*(self.dz_curr[i][0]+self.y_lin[i][0]+self.y_lin[i][-1])
        self.opti[i].subject_to(slack>=0)
        self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                      self.v_lin[i][1]+A[0][3,:]@self.dz_curr[i]+B[0][3,:]@h[:,0],
                                                      self.V_MAX))

        self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,self.a_lin[i][0]+h[0,0],self.A_MAX))



        self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN, self.df_lin[i][0]+h[1,0], self.DF_MAX))



        self.opti[i].subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[i][0]+self.a_lin[i][0]+h[0,0])*self.fps)
        self.opti[i].subject_to((-self.u_prev[i][0]+self.a_lin[i][0]+h[0,0])*self.fps<=slack+self.A_DOT_MAX)

        self.opti[i].subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[i][1]+self.df_lin[i][0]+h[1,0])*self.fps)
        self.opti[i].subject_to((-self.u_prev[i][1]+self.df_lin[i][0]+h[1,0])*self.fps<=slack+self.DF_DOT_MAX)


        if t_bar==0:

            A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
            B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
            C_block=ca.MX((4+2*N_TV)*self.N, (2+2*N_TV)*self.N)
            E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

            A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][0][0] for k in range(N_TV)])
            A_block[0:4, 4:4+2*N_TV]=B[0]@K[0]
            B_block[0:4,0:2]=B[0]
            C_block[0:4+2*N_TV,0:2+2*N_TV]=ca.diagcat(ca.vertcat(ca.DM.eye(2), ca.DM(2,2)), ca.DM.eye(2*N_TV))
            E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E[0:4,0:4], *[self.Sigma_tv_sqrt[i][k][0][0] for k in range(N_TV)])

            H=h[:,0]
            c_ev=-K[0]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][0][0], self.y_tv_ref[i][k][0][0]) for k in range(N_TV)])
            C=ca.vertcat(c_ev, *[c[k][0][0] for k in range(N_TV)])

            self.Q_tv=self.R_tv

            for t in range(1,self.N):

                oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t], self.Q_tv[i][k][0][t-1]) for k in range(N_TV)]

                for k in range(N_TV):


                    soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.DM.eye(2),ca.DM(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.DM.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
                                             slack+2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.DM.eye(2),ca.DM(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.DM.eye(2)))\
                                                    @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                                                      +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:(2+2*N_TV)*t]@C)\
                                                   +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))

                    self.opti[i].subject_to(soc_constr>0)


                A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]


                B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(2+2*N_TV):(t+1)*(2+2*N_TV)]=ca.diagcat(ca.vertcat(ca.MX.eye(2), ca.MX(2,2)), ca.MX.eye(2*N_TV))
                c_ev=-K[t]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t]) for k in range(N_TV)])
                C=ca.vertcat(C, c_ev, *[c[k][0][t] for k in range(N_TV)])


                E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
                E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.DM(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
                E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E

                H=ca.vertcat(H, h[:,t])

            nom_z_ev=ca.kron(ca.DM.eye(self.N),ca.horzcat(ca.DM.eye(4), ca.DM(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H)
            cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)

            cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.DM.eye(self.N),self.R))

            nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
            nom_df=H.reshape((2,self.N))[1,:]
            nom_diff_df=ca.diff(nom_df+self.df_ref[i][:-1],1,1)
            nom_da=H.reshape((2,self.N))[0,:]
            nom_diff_a=ca.diff(nom_da+self.a_ref[i][:-1],1,1)

            self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                      nom_dv+self.v_lin[i][1:],
                                                      self.V_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                                                      nom_df+self.df_lin[i][:-1],
                                                      self.DF_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                                                      nom_da+self.a_lin[i][:-1],
                                                      self.A_MAX))


            self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN*self.DT-slack,
                                                      nom_diff_a,
                                                      self.A_DOT_MAX*self.DT+slack))

            self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN*self.DT-slack,
                                                      nom_diff_df,
                                                      self.DF_DOT_MAX*self.DT+slack))


        elif t_bar<self.N-1:

                mode_map=list(product([*range(self.N_modes)],repeat=N_TV))
                mode_map=sorted([(sum([10**mode_map[i][j] for j in range(len(mode_map[i]))]),)+mode_map[i] for i in range(len(mode_map))])
                mode_map=[mode_map[i][1:] for i in range(len(mode_map))]
                if not self.noswitch_bl:

                    ## This considers the possibility that the TV can switch accross modes and enumerates all allowable mode sequences.
                    ## VERY SLOW, not being used.


                    seq=list(product([*range(self.N_modes**N_TV)],repeat=min(6,t_bar+1)))
                    seq=seq[:min(self.N_seq_max, (self.N_modes**N_TV)**min(6,t_bar+1))]
                    tail_seq=[[seq[j][-1]]*(self.N-min(6,t_bar+1)) for j in range(len(seq))]

                    seq=[list(seq[i])+tail_seq[i] for i in range(len(seq))]

                else:
                    seq=[(self.N+1)*[m] for m in range(self.N_modes**N_TV)]


                for s in range(len(seq)):


                    A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
                    B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
                    C_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)
                    E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

                    A_block_prev=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
                    B_block_prev=ca.MX((4+2*N_TV)*self.N, 2*self.N)
                    C_block_prev=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)
                    E_block_prev=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

                    A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    A_block[0:4, 4:4+2*N_TV]=B[0]@K[0]

                    A_block_prev[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    A_block_prev[0:4, 4:4+2*N_TV]=B[0]@K_prev[0]

                    B_block[0:4,0:2]=B[0]
                    C_block[:4+2*N_TV,0:4+2*N_TV]=ca.diagcat(ca.DM.eye(4), ca.DM.eye(2*N_TV))
                    E_block[0:4+2*N_TV, 0:4+2*N_TV]=E

                    B_block_prev[0:4,0:2]=B[0]
                    C_block_prev[:4+2*N_TV,0:4+2*N_TV]=ca.diagcat(ca.DM.eye(4), ca.DM.eye(2*N_TV))
                    E_block_prev[0:4+2*N_TV, 0:4+2*N_TV]=E


                    H=h[:,0]
                    H_prev=h_prev[:,0]
                    c_ev=-B[0]@K[0]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][0]][k]][0], self.y_tv_ref[i][k][mode_map[seq[s][0]][k]][0]) for k in range(N_TV)])
                    C=ca.vertcat(c_ev, *[c[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])

                    c_ev_prev=-B[0]@K_prev[0]@ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][0]][k]][0], self.y_tv_ref[i][k][mode_map[seq[s][0]][k]][0]) for k in range(N_TV)])
                    C_prev=ca.vertcat(c_ev_prev, *[c[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])

                    for t in range(1,self.N+1):

                        # oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]
                        # oa_ref=[self._oa_ev_ref([self.x_lin[i][t-1], self.x_lin[i][t]], [self.y_lin[i][t-1], self.y_lin[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]
                        # oa_ref=[self._oa_ev_ref([x, x], [y, y], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1]) for k in range(N_TV)]

                        # eval_oa_i.append([(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@self.Q_tv[i][k][mode_map[seq[s][t]][k]][t-1] for k in range(N_TV)])

                        R_ev=self.R_ev[i][t-1]

                        for k in range(N_TV):

                            lmbd=self.lmbd_dual_var[i][k][mode_map[seq[s][t]][k]][:,t-1].T
                            nu=self.nu_dual_var[i][k][mode_map[seq[s][t]][k]][:,t-1].T
                            lmbd_prev=self.lmbd_prev[i][k][mode_map[seq[s][t]][k]][:,t-1].T
                            nu_prev=self.nu_prev[i][k][mode_map[seq[s][t]][k]][:,t-1].T
                            R_tv=self.R_tv[i][k][mode_map[seq[s][t]][k]][t-1]
                            # pdb.set_trace()
                            z=(lmbd_prev@self.G@R_ev-nu_prev@self.G@R_tv)@ca.horzcat(ca.DM.eye(2),ca.DM(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.DM.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]\
                              +((lmbd-lmbd_prev)@self.G@R_ev-(nu-nu_prev)@self.G@R_tv)@ca.horzcat(ca.DM.eye(2),ca.DM(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.DM.eye(2)))@E_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                            y=(lmbd_prev@self.G@R_ev-nu_prev@self.G@R_tv)@ca.horzcat(ca.DM.eye(2),ca.DM(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.DM.eye(2)))\
                              @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                              +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:(4+2*N_TV)*t]@C)\
                              +((lmbd-lmbd_prev)@self.G@R_ev-(nu-nu_prev)@self.G@R_tv)@ca.horzcat(ca.DM.eye(2),ca.DM(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.DM.eye(2)))\
                              @(A_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                              +B_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H_prev+C_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:(4+2*N_TV)*t]@C_prev)\
                              -2*(lmbd+nu)@self.g+(lmbd@self.G@R_ev-nu@self.G@R_tv)@self.z_lin[i][:2, t]-.01




                            # soc_constr=ca.soc(self.tight*z,y)

                            if self.pol_mode!=1:
                                self.opti[i].subject_to(ca.norm_2(self.tight*z)<=y)
                            else:
                                t1n=self.t_1norm[i][k][mode_map[seq[s][t]][k]][:,t-1]
                                self.opti[i].subject_to(self.opti[i].bounded(-t1n,z.T,t1n))
                                self.opti[i].subject_to(self.tight*ca.sum1(t1n)<=y)


                            self.opti[i].subject_to(lmbd>=0)
                            self.opti[i].subject_to(nu>=0)
                            self.opti[i].subject_to(ca.norm_2(lmbd@self.G@R_ev-nu@self.G@R_tv)<=2)

                        if t<self.N:
                            A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]

                            A_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@A_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]


                            if t<t_bar or seq[s][t]==0:
                                A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]

                                A_block_prev[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K_prev[t]@A_block_prev[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
                                H=ca.vertcat(H, h[:,t])
                                H_prev=ca.vertcat(H_prev, h_prev[:,t])
                                c_ev=-B[t]@K[t]@(ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])\
                                        -C_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),:(4+2*N_TV)*t]@C)
                                c_ev_prev=-B[t]@K_prev[t]@(ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])\
                                        -C_block_prev[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),:(4+2*N_TV)*t]@C_prev)

                                E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
                                E_block_prev[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M_prev[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

                                # Gains=ca.horzcat(*[M[j] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))], K[t])
                                # noise=ca.diagcat(ca.kron(ca.DM.eye(t),E[0:4,0:4]), *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])
                                # soc_da=ca.soc(Gains[0,:]@noise, slack +0.5-H[-2])
                                # soc_df=ca.soc(Gains[1,:]@noise, slack+0.2-H[-1])



                            else:
                                A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t+seq[s][t]*(self.N-t_bar)]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
                                H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])
                                c_ev=-B[t]@K[t+seq[s][t]*(self.N-t_bar)]@(ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])\
                                        -C_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),:(4+2*N_TV)*t]@C)
                                E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))], ca.DM(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

                                A_block_prev[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t+seq[s][t]*(self.N-t_bar)]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
                                H_prev=ca.vertcat(H_prev, h_prev[:,t+seq[s][t]*(self.N-t_bar)])
                                c_ev_prev=-B[t]@K_prev[t+seq[s][t]*(self.N-t_bar)]@(ca.vertcat(*[ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)])\
                                        -C_block_prev[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),:(4+2*N_TV)*t]@C_prev)
                                E_block_prev[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M_prev[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))], ca.DM(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])



                                # Gains=ca.horzcat(*[M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))], K[t+seq[s][t]*(self.N-t_bar)])
                                # noise=ca.diagcat(ca.kron(ca.DM.eye(t),E[0:4,0:4]), *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])
                                # soc_da=ca.soc(Gains[0,:]@noise, slack+0.5-H[-2])
                                # soc_df=ca.soc(Gains[1,:]@noise, slack+0.2-H[-1])



                            B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                            B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                            C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                            C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(ca.DM.eye(4), ca.DM.eye(2*N_TV))

                            C=ca.vertcat(C, c_ev, *[c[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])

                            E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]

                            E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E


                            B_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                            B_block_prev[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                            C_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                            C_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(ca.DM.eye(4), ca.DM.eye(2*N_TV))

                            C_prev=ca.vertcat(C_prev, c_ev_prev, *[c[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])

                            E_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block_prev[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]

                            E_block_prev[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E


                    nom_z_ev=ca.kron(ca.DM.eye(self.N),ca.horzcat(ca.DM.eye(4), ca.DM(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H+C_block@C)
                    nom_z_err=self.z_lin[i][:,1:].reshape((-1,1))-self.z_ref[i][:,1:].reshape((-1,1))+nom_z_ev
                    nom_z_diff= ca.diff(nom_z_ev.reshape((4,-1)),1,1).reshape((-1,1))


                    cost_matrix_z=ca.diagcat(*[1**t*self.rot_costs[i][t] for t in range(self.N)])
                    cost_matrix_u=ca.kron(ca.diagcat(*[1**i for i in range(self.N-1)]),self.R)

                    nom_z_ev_i.append(nom_z_ev)
                    nom_u_ev_i.append(H.reshape((2,self.N)))
                    nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
                    nom_df=H.reshape((2,self.N))[1,:]
                    nom_diff_df=ca.diff(nom_df+self.df_lin[i][:-1],1,1)
                    nom_da=H.reshape((2,self.N))[0,:]
                    nom_diff_a=ca.diff(nom_da+self.a_lin[i][:-1],1,1)

                    nom_diff_u=ca.diff(H.reshape((2,self.N)),1,1).reshape((-1,1))/self.DT

                    cost+=RefTrajGenerator._quad_form(nom_z_err, 10*cost_matrix_z)+\
                          RefTrajGenerator._quad_form(H,ca.kron(ca.DM.eye(self.N),ca.diag([0., 0.])))+\
                          RefTrajGenerator._quad_form(nom_z_diff,1000*cost_matrix_z[:(self.N-1)*4,:(self.N-1)*4])+\
                          RefTrajGenerator._quad_form(nom_diff_u,10*cost_matrix_u)
                          #+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),1*ca.MX.eye(2)))

                    self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                              nom_dv+self.v_lin[i][1:],
                                                              self.V_MAX+slack))


                    self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                                                              nom_df+self.df_lin[i][:-1],
                                                              self.DF_MAX))


                    self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                                                              nom_da+self.a_lin[i][:-1],
                                                              self.A_MAX))


                    self.opti[i].subject_to( self.opti[i].bounded(self.A_DOT_MIN-slack,
                                                              nom_diff_a/self.DT,
                                                              self.A_DOT_MAX+slack))


                    self.opti[i].subject_to( self.opti[i].bounded(self.DF_DOT_MIN-slack,
                                                              nom_diff_df/self.DT,
                                                              self.DF_DOT_MAX+slack))



        self.opti[i].minimize( cost )
        self.nom_z_ev.append(nom_z_ev_i)
        self.nom_u_ev.append(nom_u_ev_i)
        # self.eval_oa.append(eval_oa_i)

    def solve(self, i):
        st = time.time()

        try:
            # pdb.set_trace()
            sol = self.opti[i].solve()

            # Optimal solution.
            u_control  = sol.value(self.policy[i][0][:,0])
            v_tp1      = sol.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
            is_feas     = True

            z_lin_ev   = sol.value(self.z_lin[i])
            u_lin_ev   = sol.value(self.u_lin[i])
            z_ref_ev   = sol.value(self.z_ref[i])

            N_TV=1+int(i/self.t_bar_max)
            t_bar=i-(N_TV-1)*self.t_bar_max

            if t_bar == 0 or t_bar==self.N-1:
                M_opt=[sol.value(self.policy[i][2][j]) for j in range(int((self.N-1)*self.N/2))]
                K_opt=[sol.value(self.policy[i][1][j]) for j in range(self.N)]
                h_opt=sol.value(self.policy[i][0])

            else:
                M_opt=[sol.value(self.policy[i][2][j]) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
                K_opt=[sol.value(self.policy[i][1][j]) for j in range(t_bar+(self.N_modes*N_TV)*(self.N-t_bar))]
                h_opt=sol.value(self.policy[i][0])

            lmbd_opt    = [[sol.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]
            nu_opt     = [[sol.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]

            nom_z_ev   = [sol.value(x).reshape((4,-1))+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
            nom_u_ev   = [sol.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

            z_tv_ref    = np.array([sol.value(self.x_tv_ref[i][0][0]), sol.value(self.y_tv_ref[i][0][0])])



        except:


            # Suboptimal solution (e.g. timed out).

            if self.opti[i].stats()['return_status']=='SUBOPTIMAL':
                u_control  = self.opti[i].debug.value(self.policy[i][0][:,0])
                v_tp1      = self.opti[i].debug.value(self.v_lin[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
                is_feas     = True

                z_lin_ev   = self.opti[i].debug.value(self.z_lin[i])
                u_lin_ev   = self.opti[i].debug.value(self.u_lin[i])
                z_ref_ev   = self.opti[i].debug.value(self.z_ref[i])

                nom_z_ev   = [self.opti[i].debug.value(x).reshape((4,-1))+z_lin_ev[:,1:] for x in self.nom_z_ev[i]]
                nom_u_ev   = [self.opti[i].debug.value(x)+u_lin_ev[:,:-1] for x in self.nom_u_ev[i]]

                N_TV=1+int(i/self.t_bar_max)
                t_bar=i-(N_TV-1)*self.t_bar_max

                if t_bar == 0 or t_bar==self.N-1:
                    M_opt=[self.opti[i].debug.value(self.policy[i][2][j]) for j in range(int((self.N-1)*self.N/2))]
                    K_opt=[self.opti[i].debug.value(self.policy[i][1][j]) for j in range(self.N)]
                    h_opt=self.opti[i].debug.value(self.policy[i][0])

                else:
                    M_opt=[self.opti[i].debug.value(self.policy[i][2][j]) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
                    K_opt=[self.opti[i].debug.value(self.policy[i][1][j]) for j in range(t_bar+(self.N_modes*N_TV)*(self.N-t_bar))]
                    h_opt=self.opti[i].debug.value(self.policy[i][0])

                lmbd_opt    = [[self.opti[i].debug.value(self.lmbd_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]
                nu_opt     = [[self.opti[i].debug.value(self.nu_dual_var[i][k][j]) for j in range(self.N_modes)] for k in range(N_TV)]
                z_tv_ref    = np.array([self.opti[i].debug.value(self.x_tv_ref[i][0][0]), self.opti[i].debug.value(self.y_tv_ref[i][0][0])])


            else:
                # pdb.set_trace()
                if self.v_curr> 1:
                    u_control  = np.array([self.a_brake-self.u_ref_val[0], -self.u_ref_val[1]])
                    v_tp1      = self.v_curr+self.DT*self.a_brake
                else:
                    u_control  = np.array([0., 0.])
                    v_tp1      = self.v_next

                is_feas = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_feas
             # whether the solution is optimal or not
        if not is_feas:
            sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
        else:
            sol_dict['solve_time'] = self.opti[i].stats()["t_wall_solver"]  # how long the solver took in seconds
            sol_dict['nom_z_ev']= nom_z_ev
            sol_dict['nom_u_ev']= nom_u_ev
            sol_dict['h_opt']=h_opt
            sol_dict['M_opt']=M_opt
            sol_dict['K_opt']=K_opt
            sol_dict['lmbd_opt']=lmbd_opt
            sol_dict['nu_opt']=nu_opt
            sol_dict['z_lin']   = z_lin_ev
            sol_dict['z_ref']   = z_ref_ev
            sol_dict['z_tv_ref']= z_tv_ref



        return sol_dict

    def update(self, i, update_dict):
        self._update_ev_initial_condition(i, *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_ev_rotated_costs(i, update_dict['Rs_ev'])
        self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']] )
        self._update_ev_reference(i, *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_ev_lin(i, *[update_dict[key] for key in ['x_lin', 'y_lin', 'psi_lin', 'v_lin', 'a_lin', 'df_lin']] )
        self._update_tv_preds(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']], *[update_dict[key] for key in ['mus', 'sigmas']] )
        self._update_previous_input(i, *[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self._update_tv_shapes(i, update_dict['tv_shapes'])
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
        self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]
        self.v_next=update_dict['v_ref'][1]
        self.update_dict=update_dict

        N_TV=1+int(i/self.t_bar_max)
        t_bar=i-(N_TV-1)*self.t_bar_max

        if t_bar == 0 or t_bar==self.N-1:
            for j in range(int((self.N-1)*self.N/2)):
                self.opti[i].set_value(self.M_prev[i][j], np.zeros((2,4)))
                if 'ws' in update_dict.keys():
                    self.opti[i].set_value(self.M_prev[i][j], update_dict['ws'][2][j])
                    # self.opti[i].set_initial(self.policy[i][2][j], update_dict['ws'][2][j])
            for j in range(self.N):
                self.opti[i].set_value(self.K_prev[i][j], np.zeros((2,2*N_TV)))
                if 'ws' in update_dict.keys():
                    self.opti[i].set_value(self.K_prev[i][j], update_dict['ws'][1][j])
                    # self.opti[i].set_initial(self.policy[i][1][j], update_dict['ws'][1][j])
            self.opti[i].set_value(self.h_prev[i], np.zeros((2,self.N)))
            if 'ws' in update_dict.keys():
                self.opti[i].set_value(self.h_prev[i], update_dict['ws'][0])
                # self.opti[i].set_initial(self.policy[i][0], update_dict['ws'][0])
        else:
            for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes*N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))):
                self.opti[i].set_value(self.M_prev[i][j], np.zeros((2,4)))
                if 'ws' in update_dict.keys():
                    self.opti[i].set_value(self.M_prev[i][j], update_dict['ws'][2][j])
                    # self.opti[i].set_initial(self.policy[i][2][j], update_dict['ws'][2][j])
            for j in range(t_bar+(self.N_modes*N_TV)*(self.N-t_bar)):
                self.opti[i].set_value(self.K_prev[i][j], np.zeros((2,2*N_TV)))
                if 'ws' in update_dict.keys():
                    self.opti[i].set_value(self.K_prev[i][j], update_dict['ws'][1][j])
                    # self.opti[i].set_initial(self.policy[i][1][j], update_dict['ws'][1][j])
            self.opti[i].set_value(self.h_prev[i], np.zeros((2,t_bar+(self.N_modes*N_TV)*(self.N-t_bar))))
            if 'ws' in update_dict.keys():
                self.opti[i].set_value(self.h_prev[i], update_dict['ws'][0])
                # self.opti[i].set_initial(self.policy[i][0], update_dict['ws'][0])

        for j in range(self.N_modes):
            for k in range(N_TV):
                # self.opti[i].set_initial(self.lmbd_dual_var[i][k][j],  0.5*ca.DM.ones(4,self.N))
                # self.opti[i].set_initial(self.nu_dual_var[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_value(self.lmbd_prev[i][k][j],  0.5*ca.DM.ones(4,self.N))
                self.opti[i].set_value(self.nu_prev[i][k][j],  0.5*ca.DM.ones(4,self.N))
                if 'ws' in update_dict.keys():
                    # self.opti[i].set_initial(self.lmbd_dual_var[i][k][j], update_dict['ws'][3][k][j])
                    # self.opti[i].set_initial(self.nu_dual_var[i][k][j], update_dict['ws'][4][k][j])

                    self.opti[i].set_value(self.lmbd_prev[i][k][j], update_dict['ws'][3][k][j])
                    self.opti[i].set_value(self.nu_prev[i][k][j], update_dict['ws'][4][k][j])




    def _update_ev_initial_condition(self, i, dx0, dy0, dpsi0, dvel0):
        self.opti[i].set_value(self.dz_curr[i], ca.DM([dx0, dy0, dpsi0, dvel0]))

    def _update_ev_rotated_costs(self, i, Rs_ev):
        for t in range(self.N):
            self.opti[i].set_value(self.rot_costs[i][t], ca.diagcat(Rs_ev[t].T@self.Q[:2,:2]@Rs_ev[t], self.Q[2:,2:]))
            self.opti[i].set_value(self.R_ev[i][t], Rs_ev[t])



    def _update_tv_shapes(self, i, Q_tv):
        R=self.R_tv[i]
        N_TV=1+int(i/self.t_bar_max)

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N):
                    if t==self.N-1:
                        self.opti[i].set_value(R[k][j][t], Q_tv[k][j][t-1])
                    else:
                        self.opti[i].set_value(R[k][j][t], Q_tv[k][j][t])




    def _update_tv_initial_condition(self, i, x_tv0, y_tv0):

        N_TV=1+int(i/self.t_bar_max)
        for k in range(N_TV):
            self.opti[i].set_value(self.z_tv_curr[i][:,k], ca.DM([x_tv0[k], y_tv0[k]]))

    def _update_ev_reference(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):

        self.opti[i].set_value(self.x_ref[i],   x_ref)
        self.opti[i].set_value(self.y_ref[i],   y_ref)
        self.opti[i].set_value(self.psi_ref[i], psi_ref)
        self.opti[i].set_value(self.v_ref[i],   v_ref)
        self.opti[i].set_value(self.a_ref[i],   a_ref)
        self.opti[i].set_value(self.df_ref[i],   df_ref)

    def _update_ev_lin(self, i, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):

        self.opti[i].set_value(self.x_lin[i],   x_ref)
        self.opti[i].set_value(self.y_lin[i],   y_ref)
        self.opti[i].set_value(self.psi_lin[i], psi_ref)
        self.opti[i].set_value(self.v_lin[i],   v_ref)
        self.opti[i].set_value(self.a_lin[i],   a_ref)
        self.opti[i].set_value(self.df_lin[i],   df_ref)

    def _update_tv_preds(self, i, x_tv0, y_tv0, mu_tv, sigma_tv):

        N_TV=1+int(i/self.t_bar_max)
        self._set_ATV_TV_dynamics(i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv)
        self._set_TV_ref(i, N_TV, x_tv0, y_tv0, mu_tv)

    def _update_previous_input(self, i, acc_prev, df_prev):
        self.opti[i].set_value(self.u_prev[i], [acc_prev, df_prev])


class SMPC_MMPreds_OL():

    def __init__(self,
                N            = 10,
                DT           = 0.2,
                L_F          = 1.7213,
                L_R          = 1.4987,
                V_MIN        = 0.,
                V_MAX        = 20.0,
                A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
                A_MAX      =  2.0,
                DF_MIN     = -.5,   # min/max front steer angle constraint (rad)
                DF_MAX     =  .5,
                A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
                A_DOT_MAX  =  1.5,
                DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
                DF_DOT_MAX =  0.5,
                N_modes_MAX  =  3,
                N_TV_MAX     =  1,
                N_seq_MAX    =  50,
                T_BAR_MAX    =  4,
                TIGHTENING   =  1.64,
                NOISE_STD    =  [0.1, .1, .01, .1, 0.1], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                Q =[0.1*50., 0.005*500, 1*10., 0.1*10.], # weights on x, y, and v.
                R = [10., 1000],
                fps = 20
                ):
        self.N=N
        self.DT=DT
        self.L_F=L_F
        self.L_R=L_R
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.A_MIN=A_MIN
        self.A_MAX=A_MAX
        self.DF_MIN=DF_MIN
        self.DF_MAX=DF_MAX
        self.A_DOT_MIN=A_DOT_MIN
        self.A_DOT_MAX=A_DOT_MAX
        self.DF_DOT_MIN=DF_DOT_MIN
        self.DF_DOT_MAX=DF_DOT_MAX
        self.N_modes=N_modes_MAX
        self.N_TV_max=N_TV_MAX
        self.N_seq_max=N_seq_MAX
        self.t_bar_max=T_BAR_MAX
        self.fps=fps
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.a_brake=-7.0

        self.opti=ca.Opti("conic")
        p_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-3, 'PSDTol' : 1e-3}
        s_opts_grb = {'error_on_fail':0}

        self.opti.solver("gurobi", s_opts_grb, p_opts_grb)


        self.z_ref=self.opti.parameter(4, self.N+1)
        self.z_lin=self.opti.parameter(4, self.N+1)
        self.u_prev=self.opti.parameter(2)
        self.x_ref=self.z_ref[0, :]
        self.y_ref=self.z_ref[1, :]
        self.psi_ref=self.z_ref[2, :]
        self.v_ref=self.z_ref[3, :]
        self.u_ref=self.opti.parameter(2, self.N+1)
        self.a_ref=self.u_ref[0, :]
        self.df_ref=self.u_ref[1, :]

        self.x_lin=self.z_lin[0, :]
        self.y_lin=self.z_lin[1, :]
        self.psi_lin=self.z_lin[2, :]
        self.v_lin=self.z_lin[3, :]
        self.u_lin=self.opti.parameter(2, self.N+1)
        self.a_lin=self.u_lin[0, :]
        self.df_lin=self.u_lin[1, :]
        self.dz_curr=self.opti.parameter(4)

        self.Mu_tv = [ [ self.opti.parameter(self.N, 2) for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]
        self.Sigma_tv   = [ [ [ self.opti.parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]
        self.Sigma_tv_sqrt  =  [ [ [ self.opti.parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]
        self.Q_tv           = [ [ [ self.opti.parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]
        self.rot_costs      = [self.opti.parameter(4,4) for t in range(self.N)]
        self.policy=self._return_policy_class()
        self.slacks=self.opti.variable(1)


        self._add_constraints_and_cost(self.N_TV_max)
        self.u_ref_val=np.zeros((2,1))
        self.v_next=np.array(5.)
        self._update_ev_initial_condition(0., 0., np.pi*0., 5.0 )
        self._update_ev_rotated_costs(self.N*[np.identity(2)])
        self._update_ev_reference([self.DT *5.0* (x) for x in range(self.N+1)],
                                  [self.DT *0.0* (x) for x in range(self.N+1)], (self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
        self._update_ev_lin([self.DT *5.0* (x) for x in range(self.N+1)],
                                  [self.DT *0.0* (x) for x in range(self.N+1)], (self.N+1)*[np.pi*0.], (self.N+1)*[5.0], (self.N+1)*[0.0], (self.N+1)*[0.0] )
        self._update_tv_preds( self.N_TV_max*[20*np.ones((self.N_modes, self.N, 2))], self.N_TV_max*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
        self._update_previous_input( 0., 0. )
        self._update_tv_shapes(self.N_TV_max*[self.N_modes*[self.N*[0.1*np.identity(2)]]])
        self.solve()


    def _return_policy_class(self):


        h=self.opti.variable(2, self.N)

        return h


    def _get_LTV_EV_dynamics(self):

        A=[ca.MX(4,4) for _ in range(self.N)]
        B=[ca.MX(4, 2) for _ in range(self.N)]

        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_lin[t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_lin[t])**2



            A[t][0,2]+=self.DT*(-ca.fmax(self.v_lin[t],0.01)*ca.sin(self.psi_lin[t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_lin[t]+beta))
            A[t][1,2]+=self.DT*(ca.fmax(self.v_lin[t],0.01)*ca.cos(self.psi_lin[t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_lin[t]+beta))
            A[t][2,3]+=self.DT*(1.0/self.L_R*ca.sin(beta))
            # A[t]=ca.expm(A[t])

            B[t][0,1]=self.DT*(-ca.fmax(self.v_lin[t],0.01)*ca.sin(self.psi_lin[t]+beta)*dbeta)
            B[t][1,1]=self.DT*(ca.fmax(self.v_lin[t],0.01)*ca.cos(self.psi_lin[t]+beta)*dbeta)
            B[t][2,1]=self.DT*(ca.fmax(self.v_lin[t],0.01)/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0

            E=(ca.DM.eye(4))@ca.diag(self.noise_std[0:4])


        return A,B,E

    def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv, Q):
            x_ev_avg=0.5*(x_ev[0]+x_ev[1])
            y_ev_avg=0.5*(y_ev[0]+y_ev[1])



            x_ref_ev=x_tv+(x_ev_avg-x_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
            y_ref_ev=y_tv+(y_ev_avg-y_tv)/ca.sqrt((ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)).T@Q@(ca.vertcat(x_ev_avg, y_ev_avg)-ca.vertcat(x_tv,y_tv)))
            return ca.vertcat(x_ref_ev,y_ref_ev)


    def _add_constraints_and_cost(self,  N_TV):




        [A,B,E]=self._get_LTV_EV_dynamics()
        h=self.policy
        slack=self.slacks
        cost = 10*slack@slack
        self.opti.subject_to(slack>=0)
        self.opti.subject_to( self.opti.bounded(self.V_MIN,
                                                      self.v_ref[1]+A[0][3,:]@self.dz_curr+B[0][3,:]@h[:,0],
                                                      self.V_MAX) )

        self.opti.subject_to(self.A_MIN<=self.a_ref[0]+h[0,0])
        self.opti.subject_to(self.a_ref[0]+h[0,0]<=self.A_MAX)

        self.opti.subject_to( self.DF_MIN<=self.df_ref[0]+h[1,0])
        self.opti.subject_to(self.df_ref[0]+h[1,0]<=self.DF_MAX)


        self.opti.subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[0]+self.a_ref[0]+h[0,0])*self.fps)
        self.opti.subject_to((-self.u_prev[0]+self.a_ref[0]+h[0,0])*self.fps<=slack+self.A_DOT_MAX)

        self.opti.subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[1]+self.df_ref[0]+h[1,0])*self.fps)
        self.opti.subject_to((-self.u_prev[1]+self.df_ref[0]+h[1,0])*self.fps<=slack+self.DF_DOT_MAX)

        A_block=ca.MX((4)*self.N, 4)
        B_block=ca.MX((4)*self.N, 2*self.N)
        E_block=ca.MX((4)*self.N, (4)*self.N)

        A_block[0:4, :]=A[0]
        B_block[0:4,0:2]=B[0]
        E_block[0:4, 0:4]=E
        H=h[:,0]


        for t in range(1,self.N):
            for j in range(self.N_modes):
                oa_ref=[self._oa_ev_ref([self.x_lin[t-1], self.x_lin[t]], [self.y_lin[t-1], self.y_lin[t]], self.Mu_tv[k][j][t-1,0], self.Mu_tv[k][j][t-1,1], self.Q_tv[k][j][t-1]) for k in range(N_TV)]
                # pdb.set_trace()
                for k in range(N_TV):

                    soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-self.Mu_tv[k][j][t-1,:].T).T@self.Q_tv[k][j][t-1]@ca.horzcat(ca.DM.eye(2),-ca.MX.eye(2))@ca.diagcat(E_block[(t-1)*(4):t*(4)-2,:], self.Sigma_tv_sqrt[k][j][t-1])),
                                             2*(oa_ref[k]-self.Mu_tv[k][j][t-1,:].T).T@self.Q_tv[k][j][t-1]@(self.z_lin[0:2,t]-oa_ref[k]+A_block[(t-1)*(4):t*(4)-2,:]@self.dz_curr+B_block[(t-1)*(4):t*(4)-2,:2*t]@H))


                    self.opti.subject_to(soc_constr>0)



            A_block[t*(4):(t+1)*(4),:]=A[t]@A_block[(t-1)*(4):t*(4),:]

            B_block[t*(4):(t+1)*(4),:]=A[t]@B_block[(t-1)*(4):t*(4),:]
            B_block[t*(4):t*(4)+4,t*2:(t+1)*2]=B[t]


            E_block[t*(4):(t+1)*(4),0:t*(4)]=A[t]@E_block[(t-1)*(4):t*(4),0:t*(4)]
            E_block[t*(4):(t+1)*(4),t*(4):(t+1)*(4)]=E

            H=ca.vertcat(H, h[:,t])

        nom_z_ev=A_block@self.dz_curr+B_block@H
        nom_z_diff= ca.diff(nom_z_ev.reshape((4,-1)),1,1).reshape((-1,1))
        cost_matrix=ca.kron(ca.DM.eye(self.N),self.Q)

        nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
        nom_df=H.reshape((2,self.N))[1,:]
        nom_da=H.reshape((2,self.N))[0,:]


        nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
        nom_df=H.reshape((2,self.N))[1,:]
        nom_diff_df=ca.diff(nom_df+self.df_lin[:-1],1,1)
        nom_da=H.reshape((2,self.N))[0,:]
        nom_diff_a=ca.diff(nom_da+self.a_lin[:-1],1,1)
        nom_diff_u=ca.diff(H.reshape((2,self.N)),1,1).reshape((-1,1))/self.DT


        self.opti.subject_to( self.opti.bounded(self.V_MIN,
                                                  nom_dv+self.v_lin[1:],
                                                  self.V_MAX))


        self.opti.subject_to( self.opti.bounded(self.DF_MIN,
                                                  nom_df+self.df_lin[:-1],
                                                  self.DF_MAX))


        self.opti.subject_to( self.opti.bounded(self.A_MIN,
                                                  nom_da+self.a_lin[:-1],
                                                  self.A_MAX))


        self.opti.subject_to( self.opti.bounded(self.A_DOT_MIN*self.DT-slack,
                                                  nom_diff_a,
                                                  self.A_DOT_MAX*self.DT+slack))


        self.opti.subject_to( self.opti.bounded(self.DF_DOT_MIN*self.DT-slack,
                                                  nom_diff_df,
                                                  self.DF_DOT_MAX*self.DT+slack))

        cost_matrix_z=ca.diagcat(*[1**t*self.rot_costs[t] for t in range(self.N)])
        cost_matrix_u=ca.kron(ca.diagcat(*[1**i for i in range(self.N-1)]),self.R)



        cost+=RefTrajGenerator._quad_form(nom_z_ev, 10*cost_matrix_z)+\
                  RefTrajGenerator._quad_form(H,ca.kron(ca.DM.eye(self.N),ca.diag([0, 0])))+\
                  RefTrajGenerator._quad_form(nom_z_diff,100*cost_matrix_z[:(self.N-1)*4,:(self.N-1)*4])+\
                  RefTrajGenerator._quad_form(nom_diff_u,10*cost_matrix_u)



        self.opti.minimize( cost )


    def solve(self):
        st = time.time()

        try:
            sol = self.opti.solve_limited()
            # Optimal solution.
            u_control  = sol.value(self.policy[:,0])

            v_tp1      = sol.value(self.v_lin[1]+self.dz_curr[3]+self.DT*self.policy[0,0])
            is_feas     = True

        except:

            # Suboptimal solution (e.g. timed out).
            if self.opti.stats()['return_status']=='SUBOPTIMAL':
                u_control  = self.opti.debug.value(self.policy[:,0])
                v_tp1      = self.opti.debug.value(self.v_lin[1]+self.dz_curr[3]+self.DT*self.policy[0,0])
                is_feas     = True

            else:
                if self.v_curr> 1:
                    u_control  = np.array([self.a_brake-self.u_ref_val[0], -self.u_ref_val[1]])
                    v_tp1      = self.v_curr+self.DT*self.a_brake
                else:
                    u_control  = np.array([0., 0.])
                    v_tp1      = self.v_next

                is_feas = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_feas      # whether the solution is optimal or not
        if not is_feas:
            sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
        else:
            sol_dict['solve_time'] = self.opti.stats()["t_wall_solver"]  # how long the solver took in seconds



        return sol_dict

    def update(self, update_dict):
        self._update_ev_initial_condition( *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_ev_rotated_costs(update_dict['Rs_ev'])
        self._update_ev_reference( *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_ev_lin( *[update_dict[key] for key in ['x_lin', 'y_lin', 'psi_lin', 'v_lin', 'a_lin', 'df_lin']] )
        self._update_tv_preds(  *[update_dict[key] for key in ['mus', 'sigmas']] )
        self._update_previous_input(*[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self._update_tv_shapes(update_dict['tv_shapes'])
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
        self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]        # pdb.set_trace()
        self.v_next=update_dict['v_ref'][1]

    def _update_ev_initial_condition(self, dx0, dy0, dpsi0, dvel0):
        self.opti.set_value(self.dz_curr, ca.DM([dx0, dy0, dpsi0, dvel0]))


    def _update_ev_rotated_costs(self, Rs_ev):
        for t in range(self.N):
            self.opti.set_value(self.rot_costs[t], ca.diagcat(Rs_ev[t].T@self.Q[:2,:2]@Rs_ev[t], self.Q[2:,2:]))




    def _update_tv_shapes(self,  Q_tv):
        Q=self.Q_tv
        N_TV=len(Q_tv)

        for k in range(N_TV):
            for j in range(self.N_modes):
                for t in range(self.N):
                    if t==self.N-1:
                        self.opti.set_value(Q[k][j][t], Q_tv[k][j][t-1])
                    else:
                        self.opti.set_value(Q[k][j][t], Q_tv[k][j][t])



    def _update_ev_reference(self, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):
        self.opti.set_value(self.x_ref,   x_ref)
        self.opti.set_value(self.y_ref,   y_ref)
        self.opti.set_value(self.psi_ref, psi_ref)
        self.opti.set_value(self.v_ref,   v_ref)
        self.opti.set_value(self.a_ref,   a_ref)
        self.opti.set_value(self.df_ref,   df_ref)


    def _update_ev_lin(self, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):
        self.opti.set_value(self.x_lin,   x_ref)
        self.opti.set_value(self.y_lin,   y_ref)
        self.opti.set_value(self.psi_lin, psi_ref)
        self.opti.set_value(self.v_lin,   v_ref)
        self.opti.set_value(self.a_lin,   a_ref)
        self.opti.set_value(self.df_lin,   df_ref)

    def _update_tv_preds(self, mu_tv, sigma_tv):

        for k in range(self.N_TV_max):
            for j in range(self.N_modes):
                self.opti.set_value(self.Mu_tv[k][j], mu_tv[k][j,:,:] )
                for t in range(self.N):
                    self.opti.set_value(self.Sigma_tv[k][j][t], 0.5*sigma_tv[k][j,t,:,:] )
                    e_val,e_vec= np.linalg.eigh(0.5*sigma_tv[k][j,t,:,:])
                    self.opti.set_value(self.Sigma_tv_sqrt[k][j][t], e_vec@np.diag(np.sqrt(e_val))@e_vec.T )

    def _update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])