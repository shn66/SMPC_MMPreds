import time
import casadi as ca
import numpy as np
from itertools import product
import pdb
class RefTrajGenerator():

    def __init__(self,
                 N          = 50,     # timesteps in Optimization Horizon
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
                 R = [0.000001*10, 0.00001*100.]):        # weights on inputs
                 # Q = [100., 100., 500., 1], # weights on x, y, and v.
                 # R = [1., 10.]):        # weights on inputs

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
        # First index corresponds to index of the waypoint,
        #   i.e. wp_ref[0,:] = 0th waypoint.
        # Second index selects an element from the waypoint described by [x_k, y_k, v_k],
        #   i.e. wp_ref[k,2]=v_k

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

        # self._update_initial_condition(0., 0., 0., 0.5)

        # self._update_reference([self.DT *5.0* (x+1) for x in range(self.N)],
        #                     [self.DT *5.0* (x+1) for x in range(self.N)], self.N*[np.pi*0.25],
        #                     self.N*[1.5])

        # self._update_previous_input(0., 0.)

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


class SMPC_MMPreds():

    def __init__(self,
                N            = 10,
                DT           = 0.2,
                L_F          = 1.7213,
                L_R          = 1.4987,
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
                N_modes_MAX  =  3,
                N_TV_MAX     =  1,
                N_seq_MAX    =  100,
                T_BAR_MAX    =  5,
                D_MIN        =  5.,
                TIGHTENING   =  1.28,
                # NOISE_STD    =  [.1, .1, .01, .5, 0.2], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                # Q =[0.1*50., 0.1*50, 10., 0.01*1.], # weights on x, y, and v.
                # R = [0.01*100., 0.1*1000],       # weights on inputs
                NOISE_STD    =  [.1, .1, .01, 0.1, 0.2],
                Q =[0.1*50., 0.1*50, 10., 0.1*1.], # weights on x, y, and v.
                R = [0.1*10., 0.1*100],
                NS_BL_FLAG=False
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
        self.d_min=D_MIN
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.a_brake=-7.0
        self.noswitch_bl=NS_BL_FLAG

        self.opti=[]

        self.z_ref=[]
        self.u_prev=[]
        self.x_ref=[]
        self.y_ref=[]
        self.psi_ref=[]
        self.v_ref=[]
        self.u_ref=[]
        self.a_ref=[]
        self.df_ref=[]
        self.dz_curr=[]
        self.Sigma_tv_sqrt  =  []
#         self.Mu_tv=[]
#         self.Sigma_tv=[]
        self.T_tv=[]
        self.c_tv=[]
        self.x_tv_ref=[]
        self.y_tv_ref=[]
        self.z_tv_curr=[]

        self.policy=[]
        self.slacks=[]


        p_opts_grb = {'OutputFlag': 0}
        s_opts_grb = {'error_on_fail':0}


        for i in range((self.t_bar_max)*self.N_TV_max):
            self.opti.append(ca.Opti('conic'))
            self.opti[i].solver("gurobi", s_opts_grb, p_opts_grb)
            self.z_ref.append(self.opti[i].parameter(4, self.N+1))
            self.u_ref.append(self.opti[i].parameter(2, self.N))
            self.u_prev.append(self.opti[i].parameter(2))
            self.x_ref.append(self.z_ref[i][0, :])
            self.y_ref.append(self.z_ref[i][1, :])
            self.psi_ref.append(self.z_ref[i][2, :])
            self.v_ref.append(self.z_ref[i][3, :])

            self.a_ref.append(self.u_ref[i][0, :])
            self.df_ref.append(self.u_ref[i][1, :])

            self.dz_curr.append(self.opti[i].parameter(4))
            self.slacks.append(self.opti[i].variable(1))

            N_TV=1+int(i/self.t_bar_max)
            t_bar=i-(N_TV-1)*self.t_bar_max

#             self.Mu_tv.append([[self.opti[i].parameter(2, self.N) for j in range(self.N_modes)] for k in range(N_TV)])
#             self.Sigma_tv.append([[[self.opti[i].parameter(2, 2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])

            self.T_tv.append([[[self.opti[i].parameter(2,2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])
            self.c_tv.append([[[self.opti[i].parameter(2,1) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])

            self.x_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])
            self.y_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])

            self.Sigma_tv_sqrt.append([ [ [ self.opti[i].parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(N_TV) ])

            self.z_tv_curr.append(self.opti[i].parameter(2,N_TV))

            self.policy.append(self._return_policy_class(i, N_TV, t_bar))
            self._add_constraints_and_cost(i, N_TV, t_bar)
            self.u_ref_val=np.zeros((2,1))
            self.v_next=np.array(5.)
            self._update_ev_initial_condition(i, 0., 0., np.pi*0., 5.0 )
            self._update_tv_initial_condition(i, N_TV*[20.0], N_TV*[20.0] )
            self._update_ev_reference(i, [self.DT *5.0* (x) for x in range(self.N+1)],
                                      [self.DT *0.0* (x) for x in range(self.N+1)], (self.N+1)*[np.pi*0.], (self.N+1)*[5.0], self.N*[0.0], self.N*[0.0] )
            self._update_tv_preds(i, N_TV*[20.0], N_TV*[20.0], N_TV*[20*np.ones((self.N_modes, self.N, 2))], N_TV*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
            self._update_previous_input(i, 0.0, 0.0)
            sol=self.solve(i)


    # def _return_policy_class(self, i, N_TV, t_bar):

    #     if t_bar == 0 or t_bar==self.N-1:
    #         M=[self.opti[i].variable(2, 4+2*N_TV) for j in range(int((self.N-1)*self.N/2))]
    #         h=self.opti[i].variable(2, self.N)

    #     else:
    #         M=[self.opti[i].variable(2, 4+2*N_TV) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes**N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
    #         h=self.opti[i].variable(2, t_bar+(self.N_modes**N_TV)*(self.N-t_bar))

    #     return h,M

    def _return_policy_class(self, i, N_TV, t_bar):

        if t_bar == 0 or t_bar==self.N-1:
            M=[self.opti[i].variable(2, 4) for j in range(int((self.N-1)*self.N/2))]
            K=[self.opti[i].variable(2,2*N_TV) for j in range(self.N)]
            h=self.opti[i].variable(2, self.N)

        else:
            M=[self.opti[i].variable(2, 4) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes**N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
            K=[self.opti[i].variable(2,2*N_TV) for j in range(self.N)]
            # K=[self.opti[i].variable(2,2*N_TV) for j in range(t_bar+(self.N_modes**N_TV)*(self.N-t_bar))]
            h=self.opti[i].variable(2, t_bar+(self.N_modes**N_TV)*(self.N-t_bar))

        return h,K,M

    def _set_ATV_TV_dynamics(self, i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv):


        T=self.T_tv[i]
        c=self.c_tv[i]

        for t in range(self.N):
            if t==0:
                for k in range(N_TV):
                    for j in range(self.N_modes):
                        self.opti[i].set_value(T[k][j][t], np.identity(2))
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-np.hstack((x_tv0[k],y_tv0[k])))
                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], ca.chol(sigma_tv[k][j,t,:,:]) )
            else:
                for j in range(self.N_modes):
                    for k in range(N_TV):
                        Ltp1=ca.chol(sigma_tv[k][j,t,:,:])
                        Lt=ca.chol(sigma_tv[k][j,t-1,:,:])
                        self.opti[i].set_value(self.Sigma_tv_sqrt[i][k][j][t], ca.chol(sigma_tv[k][j,t,:,:]) )
                        self.opti[i].set_value(T[k][j][t], ca.inv(Ltp1)@Lt)
                        # self.opti[i].set_value(T[k][j][t], np.identity(2))
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-ca.inv(Ltp1)@Lt@mu_tv[k][j, t-1, :])



    def _set_TV_ref(self, i, N_TV, x_tv0, y_tv0, mu_tv):
        for t in range(self.N+1):
            if t==0:
                for k in range(N_TV):
                    for j in range(self.N_modes):

                        self.opti[i].set_value(self.x_tv_ref[i][k][j][0], x_tv0[k])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][0], y_tv0[k])
            else:
                for j in range(self.N_modes):
                    for k in range(N_TV):

                        self.opti[i].set_value(self.x_tv_ref[i][k][j][t], mu_tv[k][j,t-1,0])
                        self.opti[i].set_value(self.y_tv_ref[i][k][j][t], mu_tv[k][j,t-1,1])




    def _get_LTV_EV_dynamics(self, i, N_TV):

        A=[ca.MX.eye(4) for n in range(self.N)]
        B=[ca.MX(4, 2) for n in range(self.N)]

        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_ref[i][t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_ref[i][t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_ref[i][t])**2


            A[t][0,2]+=self.DT*(-ca.fmax(self.v_ref[i][t],0.1)*ca.sin(self.psi_ref[i][t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_ref[i][t]+beta))
            A[t][1,2]+=self.DT*(ca.fmax(self.v_ref[i][t],0.1)*ca.cos(self.psi_ref[i][t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_ref[i][t]+beta))
            A[t][2,2]+=self.DT*(ca.fmax(self.v_ref[i][t],0.1)/self.L_R*ca.sin(beta))
            # A[t]=ca.expm(A[t])

            B[t][0,1]=self.DT*(-ca.fmax(self.v_ref[i][t],0.1)*ca.sin(self.psi_ref[i][t]+beta)*dbeta)
            B[t][1,1]=self.DT*(ca.fmax(self.v_ref[i][t],0.1)*ca.cos(self.psi_ref[i][t]+beta)*dbeta)
            B[t][2,1]=self.DT*(ca.fmax(self.v_ref[i][t],0.1)/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0


        E=ca.MX(4+2*N_TV, 4+2*N_TV)
        # E=(ca.MX.eye(4))@ca.diag(self.noise_std[0:4])
        E[0:4,0:4]=(ca.MX.eye(4))@ca.diag(self.noise_std[0:4])
        E[4:, 4:]=ca.MX.eye(2*N_TV)*self.noise_std[-1]

        return A,B,E

    def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv):
            x_ref_ev=x_tv+self.d_min*(0.5*(x_ev[0]+x_ev[1])-x_tv)/ca.norm_2(0.5*ca.vertcat(x_ev[0]+x_ev[1],y_ev[0]+y_ev[1])-ca.vertcat(x_tv,y_tv))
            y_ref_ev=y_tv+self.d_min*(0.5*(y_ev[0]+y_ev[1])-y_tv)/ca.norm_2(0.5*ca.vertcat(x_ev[0]+x_ev[1],y_ev[0]+y_ev[1])-ca.vertcat(x_tv,y_tv))
            # x_ref_ev=x_tv+self.d_min*((x_ev[1])-x_tv)/ca.norm_2(ca.vertcat(x_ev[1],y_ev[1])-ca.vertcat(x_tv,y_tv))
            # y_ref_ev=y_tv+self.d_min*((y_ev[1])-y_tv)/ca.norm_2(ca.vertcat(x_ev[1],y_ev[1])-ca.vertcat(x_tv,y_tv))
            return ca.vertcat(x_ref_ev,y_ref_ev)

    # def _add_constraints_and_cost(self, i, N_TV, t_bar):



    #     T=self.T_tv[i]
    #     c=self.c_tv[i]
    #     [A,B,E]=self._get_LTV_EV_dynamics(i, N_TV)
    #     [h,M]=self.policy[i]
    #     slack=self.slacks[i]
    #     cost = 1000*slack@slack
    #     self.opti[i].subject_to(slack==0)
    #     self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
    #                                                   self.v_ref[i][1]+A[0][3,:]@self.dz_curr[i]+B[0][3,:]@h[:,0],
    #                                                   self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


    #     self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
    #                                                   self.v_ref[i][1]+A[0][3,:]@self.dz_curr[i]+B[0][3,:]@h[:,0],
    #                                                   self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )

    #     self.opti[i].subject_to( self.A_MIN<=self.a_ref[i][0]+h[0,0])
    #     self.opti[i].subject_to(self.a_ref[i][0]+h[0,0]<=self.A_MAX)

    #     self.opti[i].subject_to( self.DF_MIN<=self.df_ref[i][0]+h[1,0])
    #     self.opti[i].subject_to(self.df_ref[i][0]+h[1,0]<=self.DF_MAX)


    #     self.opti[i].subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[i][0]+self.a_ref[i][0]+h[0,0])/self.DT)
    #     self.opti[i].subject_to((-self.u_prev[i][0]+self.a_ref[i][0]+h[0,0])/self.DT<=slack+self.A_DOT_MAX)

    #     self.opti[i].subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[i][1]+self.df_ref[i][0]+h[1,0])/self.DT)
    #     self.opti[i].subject_to((-self.u_prev[i][1]+self.df_ref[i][0]+h[1,0])/self.DT<=slack+self.DF_DOT_MAX)


    #     if t_bar==0:
    #         A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
    #         B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
    #         C_block=ca.MX((4+2*N_TV)*self.N, 2*N_TV*self.N)
    #         E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

    #         A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][0][0] for k in range(N_TV)])
    #         B_block[0:4,0:2]=B[0]
    #         C_block[4:4+2*N_TV,0:2*N_TV]=ca.MX.eye(2*N_TV)
    #         # E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][0][0] for k in range(N_TV)])
    #         E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
    #         H=h[:,0]
    #         C=ca.vertcat(*[c[k][0][0] for k in range(N_TV)])

    #         for t in range(1,self.N):

    #             oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t]) for k in range(N_TV)]
    #             # pdb.set_trace()
    #             for k in range(N_TV):

    #                 soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
    #                                          slack+2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
    #                                                 @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
    #                                                   +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))**2\
    #                                                +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))

    #                 self.opti[i].subject_to(soc_constr>0)

    #                 # self.opti[i].subject_to(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)\
    #                 #                         @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
    #                 #                           +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)\
    #                 #                        +self.tight*ca.norm_2(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:])\
    #                 #                        <-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))**2\
    #                 #                        +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))


    #             A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]

    #             B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
    #             B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


    #             C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
    #             C_block[t*(4+2*N_TV)+4:(t+1)*(4+2*N_TV),t*2*N_TV:(t+1)*2*N_TV]=ca.MX.eye(2*N_TV)

    #             C=ca.vertcat(C,*[c[k][0][t] for k in range(N_TV)])


    #             E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
    #             E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[M[j] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
    #             # E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][0][t] for k in range(N_TV)])
    #             E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E

    #             H=ca.vertcat(H, h[:,t])
    #         nom_z_ev=ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H)
    #         cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)

    #         # pdb.set_trace()
    #         cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),self.R))
    #         # cost+=(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
    #             #    +B_block@H+C_block@C).T@ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV))).T@ca.kron(ca.MX.eye(self.N),self.Q)@\
    #             # ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))\
    #             # @(A_block@ca.vertcat(self.dz_curr[i],*[ca.vertcat(self.x_tv_ref[i][k][0][t],self.y_tv_ref[i][k][0][t]) for k in range(N_TV)])\
    #             #    +B_block@H+C_block@C)+H.T@ca.kron(ca.MX.eye(self.N),self.R)@H
    #         # cost+=H.T@ca.kron(ca.MX.eye(self.N),self.R)@H


    #     elif t_bar<self.N-1:

    #             mode_map=list(product([*range(self.N_modes)],repeat=N_TV))
    #             mode_map=sorted([(sum([10**mode_map[i][j] for j in range(len(mode_map[i]))]),)+mode_map[i] for i in range(len(mode_map))])
    #             mode_map=[mode_map[i][1:] for i in range(len(mode_map))]
    #             if not self.noswitch_bl:
    #                 seq=list(product([*range(self.N_modes**N_TV)],repeat=min(6,t_bar+1)))
    #                 seq=seq[:min(self.N_seq_max, (self.N_modes**N_TV)**min(6,t_bar+1))]
    #                 tail_seq=[[seq[j][-1]]*(self.N-min(6,t_bar+1)) for j in range(len(seq))]
    #                 #                 pdb.set_trace()
    #                 seq=[list(seq[i])+tail_seq[i] for i in range(len(seq))]

    #             else:
    #                 seq=[self.N*[m] for m in range(self.N_modes**N_TV)]

    #             for s in range(len(seq)):

    #                 A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
    #                 B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
    #                 C_block=ca.MX((4+2*N_TV)*self.N, 2*N_TV*self.N)
    #                 E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

    #                 A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
    #                 B_block[0:4,0:2]=B[0]
    #                 C_block[4:4+2*N_TV,0:2*N_TV]=ca.MX.eye(2*N_TV)
    #                 E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
    #                 # E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
    #                 H=h[:,0]
    #                 C=ca.vertcat(*[c[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
    #                 for t in range(1,self.N):

    #                     oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)]

    #                     for k in range(N_TV):

    #                         soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
    #                                          slack+2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
    #                                                 @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
    #                                                   +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))**2\
    #                                                +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))

    #                         self.opti[i].subject_to(soc_constr>0)
    #                         # self.opti[i].subject_to(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)\
    #                         #                         @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
    #                         #                           +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)\
    #                         #                        +self.tight*ca.norm_2(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:])\
    #                         #                        <-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))**2\
    #                         #                        +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))



    #                     A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]

    #                     B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
    #                     B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


    #                     C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
    #                     C_block[t*(4+2*N_TV)+4:(t+1)*(4+2*N_TV),t*2*N_TV:(t+1)*2*N_TV]=ca.MX.eye(2*N_TV)

    #                     C=ca.vertcat(C,*[c[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])

    #                     E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
    #                     if t<t_bar or seq[s][t]==0:
    #                         H=ca.vertcat(H, h[:,t])

    #                         E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[M[j] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

    #                     else:
    #                         H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])

    #                         E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

    #                     E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E
    #                     # E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])

    #                 nom_z_ev=ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H)
    #                 cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)
    #                 cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),self.R))
    #                 # cost+=(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
    #                 #            +B_block@H).T@ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV))).T@ca.kron(ca.MX.eye(self.N),self.Q)@\
    #                 #         ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))\
    #                 #         @(A_block@ca.vertcat(self.dz_curr[i],*[ca.vertcat(self.x_tv_ref[i][k][0][t],self.y_tv_ref[i][k][0][t]) for k in range(N_TV)])\
    #                 #            +B_block@H)+H.T@ca.kron(ca.MX.eye(self.N),self.R)@H
    #                 # cost+=H.T@ca.kron(ca.MX.eye(self.N),self.R)@H


    #     self.opti[i].minimize( cost )


    def _add_constraints_and_cost(self, i, N_TV, t_bar):



        T=self.T_tv[i]
        c=self.c_tv[i]
        [A,B,E]=self._get_LTV_EV_dynamics(i, N_TV)
        [h,K,M]=self.policy[i]
        slack=self.slacks[i]
        cost = 100*slack@slack
        self.opti[i].subject_to(slack>=0)
        self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                      self.v_ref[i][1]+A[0][3,:]@self.dz_curr[i]+B[0][3,:]@h[:,0],
                                                      self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )

        self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN-slack,self.a_ref[i][0]+h[0,0],self.A_MAX+slack))


        self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN-slack, self.df_ref[i][0]+h[1,0], self.DF_MAX+slack))



        self.opti[i].subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[i][0]+self.a_ref[i][0]+h[0,0])/self.DT)
        self.opti[i].subject_to((-self.u_prev[i][0]+self.a_ref[i][0]+h[0,0])/self.DT<=slack+self.A_DOT_MAX)

        self.opti[i].subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[i][1]+self.df_ref[i][0]+h[1,0])/self.DT)
        self.opti[i].subject_to((-self.u_prev[i][1]+self.df_ref[i][0]+h[1,0])/self.DT<=slack+self.DF_DOT_MAX)


        if t_bar==0:
            A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
            B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
            C_block=ca.MX((4+2*N_TV)*self.N, 2*N_TV*self.N)
            E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

            A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][0][0] for k in range(N_TV)])
            A_block[0:4, 4:4+2*N_TV]=B[0]@K[0]
            B_block[0:4,0:2]=B[0]
            C_block[4:4+2*N_TV,0:2*N_TV]=ca.MX.eye(2*N_TV)
            # E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][0][0] for k in range(N_TV)])
            E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
            H=h[:,0]
            C=ca.vertcat(*[c[k][0][0] for k in range(N_TV)])

            for t in range(1,self.N):

                oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t]) for k in range(N_TV)]
                # pdb.set_trace()
                for k in range(N_TV):
                    # Rot_TV

                    soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
                                             2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
                                                    @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                                                      +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))**2\
                                                   +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))

                    self.opti[i].subject_to(soc_constr>0)


                A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]


                B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                C_block[t*(4+2*N_TV)+4:(t+1)*(4+2*N_TV),t*2*N_TV:(t+1)*2*N_TV]=ca.MX.eye(2*N_TV)

                C=ca.vertcat(C,*[c[k][0][t] for k in range(N_TV)])


                E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
                E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
                # E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][0][t] for k in range(N_TV)])
                E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E

                H=ca.vertcat(H, h[:,t])
            nom_z_ev=ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H)
            cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)

            # pdb.set_trace()
            cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),self.R))

            nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
            nom_df=H.reshape((2,self.N))[1,:]
            nom_da=H.reshape((2,self.N))[0,:]
            # pdb.set_trace()
            self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                              nom_dv+self.v_ref[i][1:]+slack,
                                              2*self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )
            self.opti[i].subject_to( self.opti[i].bounded(2*self.V_MIN,
                                              nom_dv+self.v_ref[i][1:]-slack,
                                              self.V_MAX))

            self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                                              nom_df+self.df_ref[i][0:]+slack,
                                              2*self.DF_MAX))
            self.opti[i].subject_to( self.opti[i].bounded(2*self.DF_MIN,
                                              nom_df+self.df_ref[i][0:]-slack,
                                              self.DF_MAX))

            self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                                              nom_da+self.a_ref[i][0:]+slack,
                                              2*self.A_MAX))
            self.opti[i].subject_to( self.opti[i].bounded(2*self.A_MIN,
                                              nom_da+self.a_ref[i][0:]-slack,
                                              self.A_MAX))


        elif t_bar<self.N-1:

                mode_map=list(product([*range(self.N_modes)],repeat=N_TV))
                mode_map=sorted([(sum([10**mode_map[i][j] for j in range(len(mode_map[i]))]),)+mode_map[i] for i in range(len(mode_map))])
                mode_map=[mode_map[i][1:] for i in range(len(mode_map))]
                if not self.noswitch_bl:
                    seq=list(product([*range(self.N_modes**N_TV)],repeat=min(6,t_bar+1)))
                    seq=seq[:min(self.N_seq_max, (self.N_modes**N_TV)**min(6,t_bar+1))]
                    tail_seq=[[seq[j][-1]]*(self.N-min(6,t_bar+1)) for j in range(len(seq))]
                    #                 pdb.set_trace()
                    seq=[list(seq[i])+tail_seq[i] for i in range(len(seq))]

                else:
                    seq=[self.N*[m] for m in range(self.N_modes**N_TV)]

                for s in range(len(seq)):

                    A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
                    B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
                    C_block=ca.MX((4+2*N_TV)*self.N, 2*N_TV*self.N)
                    E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

                    A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    A_block[0:4, 4:4+2*N_TV]=B[0]@K[0]

                    B_block[0:4,0:2]=B[0]
                    C_block[4:4+2*N_TV,0:2*N_TV]=ca.MX.eye(2*N_TV)
                    E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
                    # E_block[0:4+2*N_TV, 0:4+2*N_TV]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    H=h[:,0]
                    C=ca.vertcat(*[c[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    for t in range(1,self.N):

                        oa_ref=[self._oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)]

                        for k in range(N_TV):

                            soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
                                             2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
                                                    @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                                                      +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))**2\
                                                   +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))

                            self.opti[i].subject_to(soc_constr>0)


                        A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                        # A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
                        A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]

                        if t<t_bar or seq[s][t]==0:
                            # A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
                            H=ca.vertcat(H, h[:,t])

                            E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
                        else:
                            # A_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,4:4+2*N_TV]=B[t]@K[t+seq[s][t]*(self.N-t_bar)]@A_block[(t-1)*(4+2*N_TV)+4:t*(4+2*N_TV),4:4+2*N_TV]
                            H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])

                            E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))], ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])


                        B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                        B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                        C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                        C_block[t*(4+2*N_TV)+4:(t+1)*(4+2*N_TV),t*2*N_TV:(t+1)*2*N_TV]=ca.MX.eye(2*N_TV)

                        C=ca.vertcat(C,*[c[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])

                        E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
                        # if t<t_bar or seq[s][t]==0:
                        #     H=ca.vertcat(H, h[:,t])

                        #     E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j],ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

                        # else:
                        #     H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])

                        #     E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[ca.horzcat(M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))], ca.MX(2,2*N_TV)) for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

                        E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E
                        # E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=ca.diagcat(E, *[self.Sigma_tv_sqrt[i][k][mode_map[seq[s][t]][k]][t].T for k in range(N_TV)])

                    nom_z_ev=ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))@(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])+B_block@H)
                    cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)
                    cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),self.R))

                    nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
                    nom_df=H.reshape((2,self.N))[1,:]
                    nom_da=H.reshape((2,self.N))[0,:]
                    # pdb.set_trace()
                    self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN-slack,
                                                      nom_dv+self.v_ref[i][1:],
                                                      self.V_MAX+slack))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


                    self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN-slack,
                                                      nom_df+self.df_ref[i][0:],
                                                      self.DF_MAX+slack))


                    self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN-slack,
                                                      nom_da+self.a_ref[i][0:],
                                                      self.A_MAX+slack))
                    # self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                    #                           nom_dv+self.v_ref[i][1:]+slack,
                    #                           2*self.V_MAX))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )
                    # self.opti[i].subject_to( self.opti[i].bounded(2*self.V_MIN,
                    #                                   nom_dv+self.v_ref[i][1:]-slack,
                    #                                   self.V_MAX))

                    # self.opti[i].subject_to( self.opti[i].bounded(self.DF_MIN,
                    #                                   nom_df+self.df_ref[i][0:]+slack,
                    #                                   2*self.DF_MAX))
                    # self.opti[i].subject_to( self.opti[i].bounded(2*self.DF_MIN,
                    #                                   nom_df+self.df_ref[i][0:]-slack,
                    #                                   self.DF_MAX))

                    # self.opti[i].subject_to( self.opti[i].bounded(self.A_MIN,
                    #                                   nom_da+self.a_ref[i][0:]+slack,
                    #                                   2*self.A_MAX))
                    # self.opti[i].subject_to( self.opti[i].bounded(2*self.A_MIN,
                    #                                   nom_da+self.a_ref[i][0:]-slack,
                    #                                   self.A_MAX))



                    # self.opti[i].subject_to( self.A_MIN-slack<=self.a_ref[i][0]+h[0,0])
                    # self.opti[i].subject_to(self.a_ref[i][0]+h[0,0]<=slack+self.A_MAX)

                    # self.opti[i].subject_to( self.DF_MIN-slack<=self.df_ref[i][0]+h[1,0])
                    # self.opti[i].subject_to(self.df_ref[i][0]+h[1,0]<=slack+self.DF_MAX)
                    # self.nom_z_ev.append(nom_z_ev)
                    # self.nom_u_ev.append(H)


        self.opti[i].minimize( cost )




    def solve(self, i):
        st = time.time()
#         sol=self.opti[i].solve_limited()
#         u_control  = sol.value(self.policy[i][0][:,0])
#         v_tp1      = sol.value(self.v_ref[i][1]+self.dz_curr[i][3]+self.DT*h[0,0])
#         is_opt     = True
#         sol = self.opti[i].solve_limited()
        try:
            sol = self.opti[i].solve_limited()
            # Optimal solution.
            u_control  = sol.value(self.policy[i][0][:,0])
#             z_opt  = sol.value(self.z_dv)
#           sl_mpc = sol.value(self.sl_dv)
            v_tp1      = sol.value(self.v_ref[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
            is_opt     = True
            # pdb.set_trace()
        except:

            # Suboptimal solution (e.g. timed out).
            if self.v_curr> 1:
                u_control  = np.array([self.a_brake-self.u_ref_val[0], 0])
                v_tp1      = self.v_curr+self.DT*self.a_brake
            else:
                u_control  = np.array([self.A_MIN-self.u_ref_val[0], -self.u_ref_val[1]])
                v_tp1      = self.v_next
#       sl_mpc = self.opti.debug.value(self.sl_dv)
#             wp_ref  = self.opti[i].debug.value(self.wp_ref)
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        if not is_opt:
            sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
        else:
            sol_dict['solve_time'] = self.opti[i].stats()["t_wall_solver"]  # how long the solver took in seconds


        return sol_dict

    def update(self, i, update_dict):
        self._update_ev_initial_condition(i, *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']] )
        self._update_ev_reference(i, *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_tv_preds(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']], *[update_dict[key] for key in ['mus', 'sigmas']] )
        self._update_previous_input(i, *[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
        self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]
        self.v_next=update_dict['v_ref'][1]
        # pdb.set_trace()


    def _update_ev_initial_condition(self, i, dx0, dy0, dpsi0, dvel0):
        self.opti[i].set_value(self.dz_curr[i], ca.DM([dx0, dy0, dpsi0, dvel0]))

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
                D_MIN        =  5.,
                TIGHTENING   =  1.28,
                NOISE_STD    =  [.1, .1, .01, 0.1, 0.2],
                Q =[0.1*50., 0.1*50, 10., 0.1*1.], # weights on x, y, and v.
                R = [0.1*10., 0.1*100]
                # NOISE_STD    =  [0.1, 0.1, 0.1, .2, 0.2], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                # Q = [50., 50., 100., 1.*1.], # weights on x, y, and v.
                # R = [.1*10., .1*100.],       # weights on inputs
                # Q =[0.1*50., 0.1*50, 10., 0.1*1.], # weights on x, y, and v.
                # R = [0.001*10., 0.1*100],
                # Q = [50., 50., 100., 1.*1.], # weights on x, y, and v.
                # R = [.1*10., .1*100.]       # weights on inputs
                # Q = [50., 50., 100., 1.*1000.], # weights on x, y, and v.
                # R = [.1*1000., .1*100000.]
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
        self.d_min=D_MIN
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)
        self.a_brake=-7.0

        self.opti=ca.Opti("conic")
        p_opts_grb = {'OutputFlag': 0}
        s_opts_grb = {'error_on_fail':0}
        # s_opts_grb = {'CPUtime': 1.15}
        self.opti.solver("gurobi", s_opts_grb, p_opts_grb)


        self.z_ref=self.opti.parameter(4, self.N+1)
        self.u_prev=self.opti.parameter(2)
        self.x_ref=self.z_ref[0, :]
        self.y_ref=self.z_ref[1, :]
        self.psi_ref=self.z_ref[2, :]
        self.v_ref=self.z_ref[3, :]
        self.u_ref=self.opti.parameter(2, self.N)
        self.a_ref=self.u_ref[0, :]
        self.df_ref=self.u_ref[1, :]
        self.dz_curr=self.opti.parameter(4)

        self.Mu_tv = [ [ self.opti.parameter(self.N, 2) for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]
        self.Sigma_tv   = [ [ [ self.opti.parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]
        self.Sigma_tv_sqrt  =  [ [ [ self.opti.parameter(2, 2) for _ in range(self.N) ] for _ in range(self.N_modes) ] for _ in range(self.N_TV_max) ]

        self.policy=self._return_policy_class()
        self.slacks=self.opti.variable(1)


        self._add_constraints_and_cost(self.N_TV_max)
        self.u_ref_val=np.zeros((2,1))
        self.v_next=np.array(5.)
        self._update_ev_initial_condition(0., 0., np.pi*0., 5.0 )
        self._update_ev_reference([self.DT *5.0* (x) for x in range(self.N+1)],
                                  [self.DT *0.0* (x) for x in range(self.N+1)], (self.N+1)*[np.pi*0.], (self.N+1)*[5.0], self.N*[0.0], self.N*[0.0] )
        self._update_tv_preds( self.N_TV_max*[20*np.ones((self.N_modes, self.N, 2))], self.N_TV_max*[np.stack(self.N_modes*[self.N*[np.identity(2)]])])
        self._update_previous_input( 0., 0. )
        self.solve()


    def _return_policy_class(self):


        h=self.opti.variable(2, self.N)

        return h


    def _get_LTV_EV_dynamics(self):

        A=[ca.MX.eye(4) for n in range(self.N)]
        B=[ca.MX(4, 2) for n in range(self.N)]

        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_ref[t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_ref[t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_ref[t])**2

            # A[t]=ca.eye(4,4)

            A[t][0,2]+=self.DT*(-ca.fmax(self.v_ref[t],0.1)*ca.sin(self.psi_ref[t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_ref[t]+beta))
            A[t][1,2]+=self.DT*(ca.fmax(self.v_ref[t],0.1)*ca.cos(self.psi_ref[t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_ref[t]+beta))
            A[t][2,2]+=self.DT*(ca.fmax(self.v_ref[t],0.1)/self.L_R*ca.sin(beta))
            # A[t]=ca.expm(A[t])

            B[t][0,1]=self.DT*(-ca.fmax(self.v_ref[t],0.1)*ca.sin(self.psi_ref[t]+beta)*dbeta)
            B[t][1,1]=self.DT*(ca.fmax(self.v_ref[t],0.1)*ca.cos(self.psi_ref[t]+beta)*dbeta)
            B[t][2,1]=self.DT*(ca.fmax(self.v_ref[t],0.1)/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0

            E=(ca.MX.eye(4))@ca.diag(self.noise_std[0:4])


        return A,B,E

    def _oa_ev_ref(self, x_ev, y_ev, x_tv, y_tv):
            # x_ref_ev=x_tv+self.d_min*(0.5*(x_ev[0]+x_ev[1])-x_tv)/ca.norm_2(0.5*ca.vertcat(x_ev[0]+x_ev[1],y_ev[0]+y_ev[1])-ca.vertcat(x_tv,y_tv))
            # y_ref_ev=y_tv+self.d_min*(0.5*(y_ev[0]+y_ev[1])-y_tv)/ca.norm_2(0.5*ca.vertcat(x_ev[0]+x_ev[1],y_ev[0]+y_ev[1])-ca.vertcat(x_tv,y_tv))
            x_ref_ev=x_tv+self.d_min*((x_ev[1])-x_tv)/ca.norm_2(ca.vertcat(x_ev[1],y_ev[1])-ca.vertcat(x_tv,y_tv))
            y_ref_ev=y_tv+self.d_min*((y_ev[1])-y_tv)/ca.norm_2(ca.vertcat(x_ev[1],y_ev[1])-ca.vertcat(x_tv,y_tv))
            return ca.vertcat(x_ref_ev,y_ref_ev)

    def _add_constraints_and_cost(self,  N_TV):




        [A,B,E]=self._get_LTV_EV_dynamics()
        h=self.policy
        slack=self.slacks
        cost = 100*slack@slack
        self.opti.subject_to(slack>=0)
        self.opti.subject_to( self.opti.bounded(self.V_MIN,
                                                      self.v_ref[1]+A[0][3,:]@self.dz_curr+B[0][3,:]@h[:,0],
                                                      self.V_MAX) )

        self.opti.subject_to(self.A_MIN-slack<=self.a_ref[0]+h[0,0])
        self.opti.subject_to(self.a_ref[0]+h[0,0]<=self.A_MAX+slack)

        self.opti.subject_to( self.DF_MIN-slack<=self.df_ref[0]+h[1,0])
        self.opti.subject_to(self.df_ref[0]+h[1,0]<=self.DF_MAX+slack)


        self.opti.subject_to( self.A_DOT_MIN-slack<=(-self.u_prev[0]+self.a_ref[0]+h[0,0])/self.DT)
        self.opti.subject_to((-self.u_prev[0]+self.a_ref[0]+h[0,0])/self.DT<=slack+self.A_DOT_MAX)

        self.opti.subject_to( self.DF_DOT_MIN-slack<=(-self.u_prev[1]+self.df_ref[0]+h[1,0])/self.DT)
        self.opti.subject_to((-self.u_prev[1]+self.df_ref[0]+h[1,0])/self.DT<=slack+self.DF_DOT_MAX)

        A_block=ca.MX((4)*self.N, 4)
        B_block=ca.MX((4)*self.N, 2*self.N)
        E_block=ca.MX((4)*self.N, (4)*self.N)

        A_block[0:4, :]=A[0]
        B_block[0:4,0:2]=B[0]
        E_block[0:4, 0:4]=E
        H=h[:,0]


        for t in range(1,self.N):
            for j in range(self.N_modes):
                oa_ref=[self._oa_ev_ref([self.x_ref[t-1], self.x_ref[t]], [self.y_ref[t-1], self.y_ref[t]], self.Mu_tv[k][j][t-1,0], self.Mu_tv[k][j][t-1,1]) for k in range(N_TV)]
                # pdb.set_trace()
                for k in range(N_TV):

                    soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-self.Mu_tv[k][j][t-1,:].T).T@ca.horzcat(ca.MX.eye(2),-ca.MX.eye(2))@ca.diagcat(E_block[(t-1)*(4):t*(4)-2,:], self.Sigma_tv_sqrt[k][j][t-1])),
                                             2*(oa_ref[k]-self.Mu_tv[k][j][t-1,:].T).T@(self.z_ref[0:2,t]-oa_ref[k]+A_block[(t-1)*(4):t*(4)-2,:]@self.dz_curr+B_block[(t-1)*(4):t*(4)-2,:2*t]@H)-self.d_min**2\
                                             +(ca.norm_2(oa_ref[k]-self.Mu_tv[k][j][t-1,:].T))**2)


                    self.opti.subject_to(soc_constr>0)



            A_block[t*(4):(t+1)*(4),:]=A[t]@A_block[(t-1)*(4):t*(4),:]

            B_block[t*(4):(t+1)*(4),:]=A[t]@B_block[(t-1)*(4):t*(4),:]
            B_block[t*(4):t*(4)+4,t*2:(t+1)*2]=B[t]


            E_block[t*(4):(t+1)*(4),0:t*(4)]=A[t]@E_block[(t-1)*(4):t*(4),0:t*(4)]
            E_block[t*(4):(t+1)*(4),t*(4):(t+1)*(4)]=E

            H=ca.vertcat(H, h[:,t])

        nom_z_ev=A_block@self.dz_curr+B_block@H
        cost_matrix=ca.kron(ca.MX.eye(self.N),self.Q)

        nom_dv=nom_z_ev.reshape((4,self.N))[3,:]
        nom_df=H.reshape((2,self.N))[1,:]
        nom_da=H.reshape((2,self.N))[0,:]
        # pdb.set_trace()
        self.opti.subject_to( self.opti.bounded(self.V_MIN-slack,
                                          nom_dv+self.v_ref[1:],
                                          self.V_MAX+slack))#self.v_ref[i][0]+self.dz_curr[i][3]+1+slack) )


        self.opti.subject_to( self.opti.bounded(self.DF_MIN-slack,
                                          nom_df+self.df_ref[0:],
                                          self.DF_MAX+slack))


        self.opti.subject_to( self.opti.bounded(self.A_MIN-slack,
                                          nom_da+self.a_ref[0:],
                                          self.A_MAX+slack))




        cost+=RefTrajGenerator._quad_form(nom_z_ev, cost_matrix)+RefTrajGenerator._quad_form(H,ca.kron(ca.MX.eye(self.N),self.R))

        self.opti.minimize( cost )


    def solve(self):
        st = time.time()
#         sol=self.opti[i].solve_limited()
#         u_control  = sol.value(self.policy[i][0][:,0])
#         v_tp1      = sol.value(self.v_ref[i][1]+self.dz_curr[i][3]+self.DT*h[0,0])
#         is_opt     = True
#         sol = self.opti[i].solve_limited()
        try:
            sol = self.opti.solve_limited()
            # Optimal solution.
            u_control  = sol.value(self.policy[:,0])
#             z_opt  = sol.value(self.z_dv)
#           sl_mpc = sol.value(self.sl_dv)
            v_tp1      = sol.value(self.v_ref[1]+self.dz_curr[3]+self.DT*self.policy[0,0])
            is_opt     = True
            # pdb.set_trace()
        except:

            # Suboptimal solution (e.g. timed out).

            # pdb.set_trace()
            # u_control  = self.u_ref_val
            if self.v_curr> 8:
                u_control  = np.array([self.a_brake-self.u_ref_val[0], -self.u_ref_val[1]])
                v_tp1      = self.v_curr+self.DT*self.a_brake
            else:
                u_control  = np.array([self.A_MIN-self.u_ref_val[0], -self.u_ref_val[1]])
                v_tp1      = self.v_next
#       sl_mpc = self.opti.debug.value(self.sl_dv)
#             wp_ref  = self.opti[i].debug.value(self.wp_ref)
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        if not is_opt:
            sol_dict['solve_time'] = np.nan  # how long the solver took in seconds
        else:
            sol_dict['solve_time'] = self.opti.stats()["t_wall_solver"]  # how long the solver took in seconds



        return sol_dict

    def update(self, update_dict):
        self._update_ev_initial_condition( *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_ev_reference( *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_tv_preds(  *[update_dict[key] for key in ['mus', 'sigmas']] )
        self._update_previous_input(*[update_dict[key] for key in ['acc_prev', 'df_prev']] )
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
        self.v_curr=update_dict['dv0']+update_dict['v_ref'][0]        # pdb.set_trace()
        self.v_next=update_dict['v_ref'][1]

    def _update_ev_initial_condition(self, dx0, dy0, dpsi0, dvel0):
        self.opti.set_value(self.dz_curr, ca.DM([dx0, dy0, dpsi0, dvel0]))


    def _update_ev_reference(self, x_ref, y_ref, psi_ref, v_ref, a_ref, df_ref):
        self.opti.set_value(self.x_ref,   x_ref)
        self.opti.set_value(self.y_ref,   y_ref)
        self.opti.set_value(self.psi_ref, psi_ref)
        self.opti.set_value(self.v_ref,   v_ref)
        self.opti.set_value(self.a_ref,   a_ref)
        self.opti.set_value(self.df_ref,   df_ref)

    def _update_tv_preds(self, mu_tv, sigma_tv):

        for k in range(self.N_TV_max):
            for j in range(self.N_modes):
                self.opti.set_value(self.Mu_tv[k][j], mu_tv[k][j,:,:] )
                for t in range(self.N):
                    self.opti.set_value(self.Sigma_tv[k][j][t], sigma_tv[k][j,t,:,:] )
                    self.opti.set_value(self.Sigma_tv_sqrt[k][j][t], ca.chol(sigma_tv[k][j,t,:,:]) )

    def _update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])