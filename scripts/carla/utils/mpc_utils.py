import time
import casadi as ca
import numpy as np
from itertools import product
import pdb
class RefTrajGenerator():

	def __init__(self,
		         N          = 50,     # timesteps in Optimization Horizon
		         DT         = 0.2,    # discretization time between timesteps (s)
		         L_F        = 1.5213, # distance from CoG to front axle (m)
		         L_R        = 1.4987, # distance from CoG to rear axle (m)
		         V_MIN      = 0.0,    # min/max velocity constraint (m/s)
		         V_MAX      = 14.0,
		         A_MIN      = -3.0,   # min/max acceleration constraint (m/s^2)
		         A_MAX      =  2.0,
		         DF_MIN     = -0.5,   # min/max front steer angle constraint (rad)
		         DF_MAX     =  0.5,
		         A_DOT_MIN  = -1.5,   # min/max jerk constraint (m/s^3)
		         A_DOT_MAX  =  1.5,
		         DF_DOT_MIN = -0.5,   # min/max front steer angle rate constraint (rad/s)
		         DF_DOT_MAX =  0.5,
		         Q = [1., 1., 100., 0.1], # weights on x, y, and v.
		         R = [0.001*10., 0.001*500.]):        # weights on inputs

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
		# self.u_prev  = self.opti.parameter(2) # previous input: [u_{acc, -1}, u_{df, -1}]
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

# 		# Slack variables used to relax input rate constraints.
# 		# Matches self.u_dv in structure but timesteps range from -1, ..., N-1.
# 		self.sl_dv  = self.opti.variable(self.N , 2)

# 		self.sl_acc_dv = self.sl_dv[:,0]
# 		self.sl_df_dv  = self.sl_dv[:,1]

		'''
		(3) Problem Setup: Constraints, Cost, Initial Solve
		'''
		self._add_constraints()

		self._add_cost()

		self._update_initial_condition(0., 0., 0., 0.5)

		self._update_reference([self.DT *5.0* (x+1) for x in range(self.N)],
			                  [self.DT *5.0* (x+1) for x in range(self.N)], self.N*[np.pi*0.25],
			                  self.N*[1.5])

		# self._update_previous_input(0., 0.)

		# Ipopt with custom options: https://web.ca.org/docs/ -> see sec 9.1 on Opti stack.
		p_opts = {'expand': True}
		s_opts = {'max_cpu_time': 0.1, 'print_level': 0}
		self.opti.solver('ipopt', p_opts, s_opts)

		sol = self.solve()

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
# 		# Input Rate Bound Constraints
# 		self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT -  self.sl_acc_dv[0],
# 			                                     self.acc_dv[0] - self.u_prev[0],
# 			                                     self.A_DOT_MAX*self.DT   + self.sl_acc_dv[0]) )

# 		self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[0],
# 			                                     self.df_dv[0] - self.u_prev[1],
# 			                                     self.DF_DOT_MAX*self.DT  + self.sl_df_dv[0]) )

# 		for i in range(self.N - 1):
# 			self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT   -  self.sl_acc_dv[i+1],
# 				                                     self.acc_dv[i+1] - self.acc_dv[i],
# 				                                     self.A_DOT_MAX*self.DT   + self.sl_acc_dv[i+1]) )
# 			self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[i+1],
# 				                                     self.df_dv[i+1]  - self.df_dv[i],
# 				                                     self.DF_DOT_MAX*self.DT  + self.sl_df_dv[i+1]) )
		# Other Constraints
# 		self.opti.subject_to( 0 <= self.sl_df_dv )
# 		self.opti.subject_to( 0 <= self.sl_acc_dv )
		# e.g. things like collision avoidance or lateral acceleration bounds could go here.

	def _add_cost(self):
		def _quad_form(z, Q):
			return ca.mtimes(z, ca.mtimes(Q, z.T))

		cost = 0
		for i in range(self.N):
			cost += _quad_form(self.z_dv[i+1, :] - self.wp_ref[i,:], self.Q) # tracking cost

		for i in range(self.N ):
			cost += _quad_form(self.u_dv[i,:], self.R)  # input cost

# 		cost += (ca.sum1(self.sl_df_dv) + ca.sum1(self.sl_acc_dv))  # slack cost

		self.opti.minimize( cost )

	def solve(self):
		st = time.time()
		try:
			sol = self.opti.solve()
			# Optimal solution.
			u_opt  = sol.value(self.u_dv)
			z_opt  = sol.value(self.z_dv)
# 			sl_mpc = sol.value(self.sl_dv)
			wp_ref  = sol.value(self.wp_ref)
			is_opt = True
		except:
			# Suboptimal solution (e.g. timed out).
			u_opt  = self.opti.debug.value(self.u_dv)
			z_opt  = self.opti.debug.value(self.z_dv)
# 			sl_mpc = self.opti.debug.value(self.sl_dv)
			wp_ref  = self.opti.debug.value(self.wp_ref)
			is_opt = False

		solve_time = time.time() - st

		sol_dict = {}
		sol_dict['u_control']  = u_opt[0,:]  # control input to apply based on solution
		sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
		sol_dict['solve_time'] = solve_time  # how long the solver took in seconds
		sol_dict['u_opt']      = u_opt       # solution inputs (N by 2, see self.u_dv above)
		sol_dict['z_opt']      = z_opt       # solution states (N+1 by 4, see self.z_dv above)
# 		sol_dict['sl_mpc']     = sl_mpc      # solution slack vars (N by 2, see self.sl_dv above)
		sol_dict['wp_ref']     = wp_ref      # waypoints  (N by 4, see self.wp_ref above)

		return sol_dict

	def update(self, update_dict):
		self._update_initial_condition( *[update_dict[key] for key in ['x0', 'y0', 'psi0', 'v0']] )
		self._update_reference( *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref']] )
		# self._update_previous_input( *[update_dict[key] for key in ['acc_prev', 'df_prev']] )

		if 'warm_start' in update_dict.keys():
			# Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
			self.opti.set_initial(self.z_dv,  update_dict['warm_start']['z_ws'])
			self.opti.set_initial(self.u_dv,  update_dict['warm_start']['u_ws'])
# 			self.opti.set_initial(self.sl_dv, update_dict['warm_start']['sl_ws'])

	def _update_initial_condition(self, x0, y0, psi0, vel0):
		self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0])

	def _update_reference(self, x_ref, y_ref, psi_ref, v_ref):
		self.opti.set_value(self.x_wp,   x_ref)
		self.opti.set_value(self.y_wp,   y_ref)
		self.opti.set_value(self.psi_wp, psi_ref)
		self.opti.set_value(self.v_wp,   v_ref)





class SMPC_MMPreds():

    def __init__(self,
                N            =  8,
                DT           = 0.2,
                L_F          = 1.5213,
                L_R          = 1.4987,
                V_MIN        = 0.0,
                V_MAX        = 15.0,
                N_modes_MAX  =  2,
                N_TV_MAX     =  1,
                N_seq_MAX    =  100,
                T_BAR_MAX    =  4,
                D_MIN        =  3.,
                TIGHTENING   =  1.2,
                NOISE_STD    =  [0.2, 0.2, 0.2, 0.01, 0.2], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV]
                Q = [5., 5., 10., 1.], # weights on x, y, and v.
                R = [0.01*1., 0.01*100.]       # weights on inputs
                ):
        self.N=N
        self.DT=DT
        self.L_F=L_F
        self.L_R=L_R
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.N_modes=N_modes_MAX
        self.N_TV_max=N_TV_MAX
        self.N_seq_max=N_seq_MAX
        self.t_bar_max=T_BAR_MAX
        self.d_min=D_MIN
        self.tight=TIGHTENING
        self.noise_std=NOISE_STD
        self.Q = ca.diag(Q)
        self.R = ca.diag(R)


        self.opti=[]

        self.z_ref=[]
        self.x_ref=[]
        self.y_ref=[]
        self.psi_ref=[]
        self.v_ref=[]
        self.u_ref=[]
        self.a_ref=[]
        self.df_ref=[]
        self.dz_curr=[]

#         self.Mu_tv=[]
#         self.Sigma_tv=[]
        self.T_tv=[]
        self.c_tv=[]
        self.x_tv_ref=[]
        self.y_tv_ref=[]
        self.z_tv_curr=[]

        self.policy=[]

        p_opts = {'expand': True}
        s_opts = {'max_cpu_time': 0.15, 'print_level': 0}
        p_opts_scs = {'max_time_milliseconds': 500}
        s_opts_scs = {'verbose': False}

        for i in range((self.t_bar_max)*self.N_TV_max):
            self.opti.append(ca.Opti())
            self.opti[i].solver("ipopt", p_opts, s_opts)

            self.z_ref.append(self.opti[i].parameter(4, self.N+1))
            self.u_ref.append(self.opti[i].parameter(2, self.N))

            self.x_ref.append(self.z_ref[i][0, :])
            self.y_ref.append(self.z_ref[i][1, :])
            self.psi_ref.append(self.z_ref[i][2, :])
            self.v_ref.append(self.z_ref[i][3, :])

            self.a_ref.append(self.u_ref[i][0, :])
            self.df_ref.append(self.u_ref[i][1, :])

            self.dz_curr.append(self.opti[i].parameter(4))


            N_TV=1+int(i/self.t_bar_max)
            t_bar=i-(N_TV-1)*self.t_bar_max

#             self.Mu_tv.append([[self.opti[i].parameter(2, self.N) for j in range(self.N_modes)] for k in range(N_TV)])
#             self.Sigma_tv.append([[[self.opti[i].parameter(2, 2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])

            self.T_tv.append([[[self.opti[i].parameter(2,2) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])
            self.c_tv.append([[[self.opti[i].parameter(2,1) for n in range(self.N)] for j in range(self.N_modes)] for k in range(N_TV)])

            self.x_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])
            self.y_tv_ref.append([[self.opti[i].parameter(1,self.N+1) for j in range(self.N_modes)] for k in range(N_TV)])

#             self.x_tv_ref.append([[self.Mu_tv[i][k][j][0,:] for j in range(self.N_modes)] for k in range(N_TV)])
#             self.y_tv_ref.append([[self.Mu_tv[i][k][j][1,:] for j in range(self.N_modes)] for k in range(N_TV)])


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

            sol=self.solve(i)


    def _return_policy_class(self, i, N_TV, t_bar):

        if t_bar == 0 or t_bar==self.N-1:
            M=[self.opti[i].variable(2, 4+2*N_TV) for j in range(int((self.N-1)*self.N/2))]
            h=self.opti[i].variable(2, self.N)

        else:
            M=[self.opti[i].variable(2, 4+2*N_TV) for j in range(int((t_bar-1)*t_bar/2)+(self.N_modes**N_TV)*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2)))]
            h=self.opti[i].variable(2, t_bar+(self.N_modes**N_TV)*(self.N-t_bar))

        return h,M

    def _set_ATV_TV_dynamics(self, i, N_TV, x_tv0, y_tv0, mu_tv, sigma_tv):


        T=self.T_tv[i]
        c=self.c_tv[i]

        for t in range(self.N):
            if t==0:
                for k in range(N_TV):
                    for j in range(self.N_modes):
                        self.opti[i].set_value(T[k][j][t], np.identity(2))
                        self.opti[i].set_value(c[k][j][t], mu_tv[k][j, t, :]-np.hstack((x_tv0[k],y_tv0[k])))
            else:
                for j in range(self.N_modes):
                    for k in range(N_TV):
                        Ltp1=ca.chol(sigma_tv[k][j,t,:,:])
                        Lt=ca.chol(sigma_tv[k][j,t-1,:,:])

                        self.opti[i].set_value(T[k][j][t], ca.inv(Ltp1)@Lt)
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

        A=[ca.MX(4, 4) for n in range(self.N)]
        B=[ca.MX(4, 2) for n in range(self.N)]

        for t in range(self.N):
            beta = ca.atan( self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_ref[i][t]) )
            dbeta = self.L_R/(1+(self.L_R / (self.L_F + self.L_R) * ca.tan(self.df_ref[i][t]))**2)/(self.L_R+self.L_F)/ca.cos(self.df_ref[i][t])**2

            A[t]=ca.MX.eye(4)

            A[t][0,2]+=self.DT*(-self.v_ref[i][t]*ca.sin(self.psi_ref[i][t]+beta))
            A[t][0,3]+=self.DT*(ca.cos(self.psi_ref[i][t]+beta))
            A[t][1,2]+=self.DT*(self.v_ref[i][t]*ca.cos(self.psi_ref[i][t]+beta))
            A[t][1,3]+=self.DT*(ca.sin(self.psi_ref[i][t]+beta))
            A[t][2,2]+=self.DT*(self.v_ref[i][t]/self.L_R*ca.sin(beta))

            B[t][0,1]=self.DT*(-self.v_ref[i][t]*ca.sin(self.psi_ref[i][t]+beta)*dbeta)
            B[t][1,1]=self.DT*(self.v_ref[i][t]*ca.cos(self.psi_ref[i][t]+beta)*dbeta)
            B[t][2,1]=self.DT*(self.v_ref[i][t]/self.L_R*ca.cos(beta)*dbeta)
            B[t][3,0]=self.DT*1.0


        E=ca.MX(4+2*N_TV, 4+2*N_TV)
        E[0:4,0:4]=(ca.MX.eye(4))@ca.diag(self.noise_std[0:4])
        E[4:, 4:]=ca.MX.eye(2*N_TV)*self.noise_std[-1]

        return A,B,E

    def _add_constraints_and_cost(self, i, N_TV, t_bar):

        def _oa_ev_ref(x_ev, y_ev, x_tv, y_tv):
            x_ref_ev=x_tv+self.d_min*(0.5*(x_ev[0]+x_ev[1])-x_tv)/ca.norm_2(0.5*ca.vertcat(x_ev[0]+x_ev[1],y_ev[0]+y_ev[1])-ca.vertcat(x_tv,y_tv))
            y_ref_ev=y_tv+self.d_min*(0.5*(y_ev[0]+y_ev[1])-y_tv)/ca.norm_2(0.5*ca.vertcat(x_ev[0]+x_ev[1],y_ev[0]+y_ev[1])-ca.vertcat(x_tv,y_tv))
            return ca.vertcat(x_ref_ev,y_ref_ev)

        T=self.T_tv[i]
        c=self.c_tv[i]
        [A,B,E]=self._get_LTV_EV_dynamics(i, N_TV)
        [h,M]=self.policy[i]

        cost = 0

        self.opti[i].subject_to( self.opti[i].bounded(self.V_MIN,
                                                      self.v_ref[i][1]+A[0][3,:]@self.dz_curr[i]+B[0][3,:]@h[:,0],
                                                      self.V_MAX) )
        if t_bar==0:
            A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
            B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
            C_block=ca.MX((4+2*N_TV)*self.N, 2*N_TV*self.N)
            E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

            A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][0][0] for k in range(N_TV)])
            B_block[0:4,0:2]=B[0]
            C_block[4:4+2*N_TV,0:2*N_TV]=ca.MX.eye(2*N_TV)
            E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
            H=h[:,0]
            C=ca.vertcat(*[c[k][0][0] for k in range(N_TV)])

            for t in range(1,self.N):

                oa_ref=[_oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t]) for k in range(N_TV)]

                # for k in range(N_TV):

#                     soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
#                                              2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
#                                                     @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
#                                                       +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))**2\
#                                                    +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))

#                     self.opti[i].subject_to(soc_constr>0)

                    # self.opti[i].subject_to(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)\
                    #                         @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                    #                           +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)\
                    #                        +self.tight*ca.norm_2(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:])\
                    #                        <-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))**2\
                    #                        +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][0][t], self.y_tv_ref[i][k][0][t])))


                A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]

                B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                C_block[t*(4+2*N_TV)+4:(t+1)*(4+2*N_TV),t*2*N_TV:(t+1)*2*N_TV]=ca.MX.eye(2*N_TV)

                C=ca.vertcat(C,*[c[k][0][t] for k in range(N_TV)])


                E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=ca.diagcat(A[t], *[T[k][0][t] for k in range(N_TV)])@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
                E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[M[j] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])
                E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E

                H=ca.vertcat(H, h[:,t])

            cost+=(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                   +B_block@H+C_block@C).T@ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV))).T@ca.kron(ca.MX.eye(self.N),self.Q)@\
                ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))\
                @(A_block@ca.vertcat(self.dz_curr[i],*[ca.vertcat(self.x_tv_ref[i][k][0][t],self.y_tv_ref[i][k][0][t]) for k in range(N_TV)])\
                   +B_block@H+C_block@C)+H.T@ca.kron(ca.MX.eye(self.N),self.R)@H


        elif t_bar<self.N-1:
                seq=list(product([*range(self.N_modes**N_TV)],repeat=min(6,t_bar+1)))
                seq=seq[:min(self.N_seq_max, (self.N_modes**N_TV)**min(6,t_bar+1))]
                tail_seq=[[seq[j][-1]]*(self.N-min(6,t_bar+1)) for j in range(len(seq))]
#                 pdb.set_trace()
                seq=[list(seq[i])+tail_seq[i] for i in range(len(seq))]

                mode_map=list(product([*range(self.N_modes)],repeat=N_TV))
                mode_map=sorted([(sum([10**mode_map[i][j] for j in range(len(mode_map[i]))]),)+mode_map[i] for i in range(len(mode_map))])
                mode_map=[mode_map[i][1:] for i in range(len(mode_map))]

                for s in range(len(seq)):

                    A_block=ca.MX((4+2*N_TV)*self.N, 4+2*N_TV)
                    B_block=ca.MX((4+2*N_TV)*self.N, 2*self.N)
                    C_block=ca.MX((4+2*N_TV)*self.N, 2*N_TV*self.N)
                    E_block=ca.MX((4+2*N_TV)*self.N, (4+2*N_TV)*self.N)

                    A_block[0:4+2*N_TV, :]=ca.diagcat(A[0], *[T[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    B_block[0:4,0:2]=B[0]
                    C_block[4:4+2*N_TV,0:2*N_TV]=ca.MX.eye(2*N_TV)
                    E_block[0:4+2*N_TV, 0:4+2*N_TV]=E
                    H=h[:,0]
                    C=ca.vertcat(*[c[k][mode_map[seq[s][0]][k]][0] for k in range(N_TV)])
                    for t in range(1,self.N):

                        oa_ref=[_oa_ev_ref([self.x_ref[i][t-1], self.x_ref[i][t]], [self.y_ref[i][t-1], self.y_ref[i][t]], self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t]) for k in range(N_TV)]

                        # for k in range(N_TV):

#                             soc_constr=ca.soc(self.tight*(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]),
#                                              2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)))\
#                                                     @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
#                                                       +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))**2\
#                                                    +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))

#                             self.opti[i].subject_to(soc_constr>0)
                            # self.opti[i].subject_to(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)\
                            #                         @(A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                            #                           +B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*t]@H+C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:2*N_TV*t]@C)\
                            #                        +self.tight*ca.norm_2(-2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@ca.horzcat(ca.MX.eye(2),ca.MX(2,2),ca.kron([-int(j==k) for j in range(N_TV)],ca.MX.eye(2)).T)@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:])\
                            #                        <-self.d_min**2+(ca.norm_2(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))**2\
                            #                        +2*(oa_ref[k]-ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])).T@(self.z_ref[i][0:2,t]-oa_ref[k]+ca.vertcat(self.x_tv_ref[i][k][mode_map[seq[s][t]][k]][t], self.y_tv_ref[i][k][mode_map[seq[s][t]][k]][t])))



                        A_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@A_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]

                        B_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@B_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                        B_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,t*2:(t+1)*2]=B[t]


                        C_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),:]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@C_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),:]
                        C_block[t*(4+2*N_TV)+4:(t+1)*(4+2*N_TV),t*2*N_TV:(t+1)*2*N_TV]=ca.MX.eye(2*N_TV)

                        C=ca.vertcat(C,*[c[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])

                        E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),0:t*(4+2*N_TV)]=ca.diagcat(A[t], *[T[k][mode_map[seq[s][t]][k]][t] for k in range(N_TV)])@E_block[(t-1)*(4+2*N_TV):t*(4+2*N_TV),0:t*(4+2*N_TV)]
                        if t<t_bar or seq[s][t]==0:
                            H=ca.vertcat(H, h[:,t])

                            E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[M[j] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

                        else:
                            H=ca.vertcat(H, h[:,t+seq[s][t]*(self.N-t_bar)])

                            E_block[t*(4+2*N_TV):t*(4+2*N_TV)+4,0:t*(4+2*N_TV)]+=B[t]@ca.horzcat(*[M[j+seq[s][t]*(int((self.N-1)*self.N/2)-int((t_bar-1)*t_bar/2))] for j in range(int(t*(t-1)/2),int(t*(t+1)/2))])

                        E_block[t*(4+2*N_TV):(t+1)*(4+2*N_TV),t*(4+2*N_TV):(t+1)*(4+2*N_TV)]=E

                    cost+=(A_block@ca.vertcat(self.dz_curr[i],*[self.z_tv_curr[i][:,k] for k in range(N_TV)])\
                               +B_block@H+C_block@C).T@ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV))).T@ca.kron(ca.MX.eye(self.N),self.Q)@\
                            ca.kron(ca.MX.eye(self.N),ca.horzcat(ca.MX.eye(4), ca.MX(4,2*N_TV)))\
                            @(A_block@ca.vertcat(self.dz_curr[i],*[ca.vertcat(self.x_tv_ref[i][k][0][t],self.y_tv_ref[i][k][0][t]) for k in range(N_TV)])\
                               +B_block@H+C_block@C)+H.T@ca.kron(ca.MX.eye(self.N),self.R)@H


        self.opti[i].minimize( cost )




    def solve(self, i):
        st = time.time()
#         sol=self.opti[i].solve_limited()
#         u_control  = sol.value(self.policy[i][0][:,0])
#         v_tp1      = sol.value(self.v_ref[i][1]+self.dz_curr[i][3]+self.DT*h[0,0])
#         is_opt     = True
#         sol = self.opti[i].solve_limited()
        try:
            sol = self.opti[i].solve()
            # Optimal solution.
            u_control  = sol.value(self.policy[i][0][:,0])
#             z_opt  = sol.value(self.z_dv)
# 			sl_mpc = sol.value(self.sl_dv)
            v_tp1      = sol.value(self.v_ref[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
            is_opt     = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_control  = self.opti[i].debug.value(self.policy[i][0][:,0])
            # u_control  = self.u_ref_val
            v_tp1      = self.opti[i].debug.value(self.v_ref[i][1]+self.dz_curr[i][3]+self.DT*self.policy[i][0][0,0])
            # v_tp1      = self.v_next
# 		sl_mpc = self.opti.debug.value(self.sl_dv)
#             wp_ref  = self.opti[i].debug.value(self.wp_ref)
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['v_next']     = v_tp1
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        sol_dict['solve_time'] = solve_time  # how long the solver took in seconds


        return sol_dict

    def update(self, i, update_dict):
        self._update_ev_initial_condition(i, *[update_dict[key] for key in ['dx0', 'dy0', 'dpsi0', 'dv0']] )
        self._update_tv_initial_condition(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']] )
        self._update_ev_reference(i, *[update_dict[key] for key in ['x_ref', 'y_ref', 'psi_ref', 'v_ref', 'a_ref', 'df_ref']] )
        self._update_tv_preds(i, *[update_dict[key] for key in ['x_tv0', 'y_tv0']], *[update_dict[key] for key in ['mus', 'sigmas']] )
        self.u_ref_val=np.hstack((update_dict['a_ref'][0],update_dict['df_ref'][0]))
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