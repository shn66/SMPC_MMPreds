#!/usr/bin python3

import numpy as np
import scipy.linalg as la
import scipy.signal

import casadi as ca

import array
from typing import Tuple

from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from mpclab_common.models.abstract_model import AbstractModel

#Should be changed
from mpclab_common.models.model_types import ModelConfig
from dataclasses import dataclass, field


@dataclass
class EnergyConfig(ModelConfig):
    dt: float                       = field(default = 0.1)   # interval of an entire simulation step
    discretization_method: str      = field(default = 'euler')
    M: int                          = field(default = 10) # RK4 integration steps
    # Ioniq PHEV
    mass: float                     = field(default = 1500)
    gravity: float                  = field(default = 9.81)

class CasadiArterialRoadDrivingModel(AbstractModel):
    '''
    Base class for dynamics models that use casadi for their models.
    Implements common functions for linearizing models, integrating models, etc...
    '''
    def __init__(self, model_config):
        super().__init__(model_config)

    def precompute_model(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q:  ca.MX with elements of state vector q
        self.sym_u:  ca.MX with elements of control vector u
        self.sym_dq: ca.MX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''
        dyn_inputs = [self.sym_q, self.sym_u]

        self.fc = ca.Function('fc', dyn_inputs, [self.sym_dq], self.options('fc'))

        # symbolic jacobians
        self.sym_Ac = ca.jacobian(self.sym_dq, self.sym_q)
        self.sym_Bc = ca.jacobian(self.sym_dq, self.sym_u)
        self.sym_Cc = self.sym_dq

        # CasADi functions for Jacobians
        self.fA = ca.Function('fA', dyn_inputs, [self.sym_Ac], self.options('fA'))
        self.fB = ca.Function('fB', dyn_inputs, [self.sym_Bc], self.options('fB'))
        self.fC = ca.Function('fC', dyn_inputs, [self.sym_Cc], self.options('fC'))

        # Discretization
        if self.model_config.discretization_method == 'euler':
            sym_q_kp1 = self.sym_q + self.dt*self.fc(*dyn_inputs)
        elif self.model_config.discretization_method == 'rk4':
            sym_q_kp1 = self.f_d_rk4(*dyn_inputs)
        else:
            raise ValueError('Discretization method of %s not recognized' % self.model_config.discretization_method)


        self.fd = ca.Function('fd', dyn_inputs, [sym_q_kp1], self.options('fd'))

        self.sym_Ad = ca.jacobian(sym_q_kp1, self.sym_q)
        self.sym_Bd = ca.jacobian(sym_q_kp1, self.sym_u)
        self.sym_Cd = sym_q_kp1

        self.fAd = ca.Function('fAd', dyn_inputs, [self.sym_Ad], self.options('fAd'))
        self.fBd = ca.Function('fBd', dyn_inputs, [self.sym_Bd], self.options('fBd'))
        self.fCd = ca.Function('fCd', dyn_inputs, [self.sym_Cd], self.options('fCd'))

        return

    def local_discretization(self, vehicle_state: VehicleState, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Local Euler discretization of continuous time dynamics
        x_{k+1} = A x_k + B u_k + C
        '''
        q, u = self.state2qu(vehicle_state)  # TODO state2qu
        args = [q, u]

        A = np.eye(q.shape[0]) + dt * np.array(self.fA(*args))
        B = dt * np.array(self.fB(*args))
        C = q + dt * np.array(self.fC(*args)).squeeze() - A @ q - B @ u

        if C.ndim == 1:
            C = np.expand_dims(C,1)
        return A, B, C

    def step(self, vehicle_state: VehicleState,
            method: str = 'RK45'):
        '''
        steps noise-free model forward one time step (self.dt) using numerical integration
        '''
        q, u = self.state2qu(vehicle_state) # TODO state2qu

        f = lambda t, qs: (self.fc(qs, u)).toarray().squeeze()

        t = vehicle_state.t - self.t0
        tf = t + self.dt
        q_n = scipy.integrate.solve_ivp(f, [t,tf], q, method = method).y[:,-1]

        self.qu2state(vehicle_state, q_n, u) # TODO qu2state
        vehicle_state.t = tf + self.t0
        vehicle_state.a.a_long, vehicle_state.a.a_tran = self.faccel(q_n, u)

    def f_d_rk4(self, x, u):
        '''
        Discrete nonlinear dynamics (RK4 approx.)
        '''
        x_p = x
        for i in range(self.M):
            a1 = self.fc(x_p, u)
            a2 = self.fc(x_p+(self.h/2)*a1, u)
            a3 = self.fc(x_p+(self.h/2)*a2, u)
            a4 = self.fc(x_p+self.h*a3, u)
            x_p += self.h*(a1 + 2*a2 + 2*a3 + a4)/6
        return x_p


class CasadiARPAE(CasadiArterialRoadDrivingModel):
    '''
    Global frame of reference kinematic bicycle
    Body frame velocities and global frame positions
    '''
    def __init__(self, model_config): #, sym_state, sym_control):
        super().__init__(model_config)
        self.dt = self.model_config.dt
        self.M = self.model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

        self.L_f = 1.4
        self.L_r = 1.4
        self.m = self.model_config.mass
        self.g = self.model_config.gravity
        self.cd = 0.0 # air drag

        self.n_q = 4 # num of state
        self.n_u = 2 # num of control

        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.p.s, state.pt.ds, state.p.x_tran, state.p.e_psi])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.p.s, state.pt.ds, state.p.x_tran, state.p.e_psi])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.p.s      = q[0]
        state.pt.ds      = q[1]
        state.p.x_tran = q[2]
        state.p.e_psi    = q[3]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.p.s      = q[0]
            state.pt.ds      = q[1]
            state.p.x_tran = q[2]
            state.p.e_psi    = q[3]
            # if u is not None:
            #     state.w.w_psi = q[2] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
            #     state.v.v_tran = state.w.w_psi * self.L_r
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        # if u is not None:
        #     psidot = np.multiply(q[:-1,0]*self.L_r, np.sin(np.arctan(np.tan(u[:,1])*self.L_f/(self.L_f + self.L_r))))
        #     psidot = np.append(psidot, psidot[-1])
        #     v_tran = psidot*self.L_r

        if q is not None:
            prediction.s      = array.array('d', q[:,0])
            prediction.v_long      = array.array('d', q[:,1])
            prediction.x_tran = array.array('d', q[:,2])
            prediction.e_psi    = array.array('d', q[:,3])
            # if u is not None:
            #     prediction.psidot = array.array('d', psidot)
            #     prediction.v_tran = array.array('d', v_tran)
        if u is not None:
            prediction.u_a = array.array('d', u[:,0])
            prediction.u_steer = array.array('d', u[:,1])

    def modelconstraint(self, road_curvature, sym_state, sym_control):
        '''
        wraps up model initialization
        Alternative to precompute_model.
        (1) because our one is LTV model.
        (2) Casadi nlp cannot handle ca.Function type constraints
        require the following fields to be initialized:
        self.sym_q:  ca.MX with elements of state vector q
        self.sym_u:  ca.MX with elements of control vector u
        self.sym_dq: ca.MX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''

        # state and state derivative functions
        self.sym_q  = sym_state
        self.sym_u  = sym_control

        self.sym_s = self.sym_q[0]
        self.sym_sdot = self.sym_q[1]
        self.sym_y = self.sym_q[2]
        self.sym_e_psi = self.sym_q[3]

        self.sym_u_a = self.sym_u[0]
        self.sym_u_steer = self.sym_u[1]

        # time derivatives
        self.sym_ds = self.sym_sdot
        # self.sym_ds = self.sym_sdot * ca.cos(self.sym_e_psi) #/ (1 - self.sym_y*road_curvature)
        self.sym_dsdot = self.sym_u_a - 1/self.m * self.cd * self.sym_sdot**2
        self.sym_dy = self.sym_sdot * ca.sin(self.sym_e_psi)
        self.sym_de_psi = self.sym_u_steer - road_curvature * self.sym_sdot * ca.cos(self.sym_e_psi)

        # pos also depends on acc (s = s0 + vt + 1/2at^2)
        self.sym_dq = ca.vertcat(self.sym_ds+0.5*self.sym_dsdot*self.dt, self.sym_dsdot, self.sym_dy, self.sym_de_psi)
        # self.sym_dq = ca.vertcat(self.sym_ds, self.sym_dsdot, self.sym_dy, self.sym_de_psi)

        sym_q_kp1 = self.sym_q + self.dt*self.sym_dq

        return sym_q_kp1



class CasadiARPAE_w_delay(CasadiArterialRoadDrivingModel):
    '''
    Global frame of reference kinematic bicycle
    Body frame velocities and global frame positions
    '''
    def __init__(self, model_config): #, sym_state, sym_control):
        super().__init__(model_config)
        self.dt = self.model_config.dt
        self.M = self.model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

        self.L_f = 1.4
        self.L_r = 1.4
        self.m = self.model_config.mass
        self.g = self.model_config.gravity
        self.cd = 28 # air drag

        self.n_q = 6 # num of state
        self.n_u = 2 # num of control

        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.p.s, state.pt.ds, state.p.x_tran, state.p.e_psi, state.u.u_a, state.u.u_steer])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.p.s, state.pt.ds, state.p.x_tran, state.p.e_psi, state.u.u_a, state.u.u_steer])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.p.s      = q[0]
        state.pt.ds      = q[1]
        state.p.x_tran = q[2]
        state.p.e_psi    = q[3]
        state.u.u_a = q[4]
        state.u.u_steer = q[5]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.p.s      = q[0]
            state.pt.ds      = q[1]
            state.p.x_tran = q[2]
            state.p.e_psi    = q[3]
            state.u.u_a = q[4]
            state.u.u_steer = q[5]
            # if u is not None:
            #     state.w.w_psi = q[2] / self.L_r * np.sin(np.arctan(np.tan(u[1]) * self.L_f / (self.L_f + self.L_r)))
            #     state.v.v_tran = state.w.w_psi * self.L_r
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        # if u is not None:
        #     psidot = np.multiply(q[:-1,0]*self.L_r, np.sin(np.arctan(np.tan(u[:,1])*self.L_f/(self.L_f + self.L_r))))
        #     psidot = np.append(psidot, psidot[-1])
        #     v_tran = psidot*self.L_r

        if q is not None:
            prediction.s      = array.array('d', q[:,0])
            prediction.v_long      = array.array('d', q[:,1])
            prediction.x_tran = array.array('d', q[:,2])
            prediction.e_psi    = array.array('d', q[:,3])
            # if u is not None:
            #     prediction.psidot = array.array('d', psidot)
            #     prediction.v_tran = array.array('d', v_tran)
        if u is not None:
            prediction.u_a = array.array('d', u[:,0])
            prediction.u_steer = array.array('d', u[:,1])

    def modelconstraint(self, road_curvature, sym_state, sym_control):
        '''
        wraps up model initialization
        Alternative to precompute_model.
        (1) because our one is LTV model.
        (2) Casadi nlp cannot handle ca.Function type constraints
        require the following fields to be initialized:
        self.sym_q:  ca.MX with elements of state vector q
        self.sym_u:  ca.MX with elements of control vector u
        self.sym_dq: ca.MX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''
        # delay coefficient
        rho_x = 1 / 0.15
        rho_y = 1 / 0.2

        # state and state derivative functions
        self.sym_q  = sym_state
        self.sym_u  = sym_control

        self.sym_s = self.sym_q[0]
        self.sym_sdot = self.sym_q[1]
        self.sym_y = self.sym_q[2]
        self.sym_e_psi = self.sym_q[3]
        self.sym_delayed_u_a = self.sym_q[4]
        self.sym_delayed_u_steer = self.sym_q[5]

        self.sym_u_a = self.sym_u[0]
        self.sym_u_steer = self.sym_u[1]

        # time derivatives
        self.sym_ds = self.sym_sdot
        self.sym_dsdot = self.sym_delayed_u_a - 1/self.m * self.cd * self.sym_sdot
        self.sym_dy = self.sym_sdot * ca.sin(self.sym_e_psi) #/(1 - self.sym_y*road_curvature)
        self.sym_de_psi = self.sym_delayed_u_steer - road_curvature * self.sym_sdot * ca.cos(self.sym_e_psi)
        self.sym_d_delayed_u_a = - rho_x * self.sym_delayed_u_a + rho_x * self.sym_u_a
        self.sym_d_delayed_u_steer = - rho_y * self.sym_delayed_u_steer + rho_y * self.sym_u_steer

        # pos also depends on acc (s = s0 + vt + 1/2at^2)
        self.sym_dq = ca.vertcat(self.sym_ds+0.5*self.sym_dsdot*self.dt, self.sym_dsdot, self.sym_dy, self.sym_de_psi, self.sym_d_delayed_u_a, self.sym_d_delayed_u_steer)
        # self.sym_dq = ca.vertcat(self.sym_ds, self.sym_dsdot, self.sym_dy, self.sym_de_psi)

        sym_q_kp1 = self.sym_q + self.dt*self.sym_dq

        return sym_q_kp1



class LaneChangeKinematicModel(CasadiArterialRoadDrivingModel):
    '''
    Vehicle kinematic model for lateral motion planning
    '''
    def __init__(self, model_config): #, sym_state, sym_control):
        super().__init__(model_config)
        self.dt = 0.1
        self.M = self.model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

        self.L_f = 1.4
        self.L_r = 1.4
        self.m = self.model_config.mass
        self.g = self.model_config.gravity

        self.n_q = 3 # num of state : [station(s), lateral error(ey), heading error(ephi)]
        self.n_u = 2 # num of control : [longitudinal speed[vx], curvature[k]]

        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.p.s, state.p.x_tran, state.p.e_psi])
        u = np.array([state.u.u_a, state.u.u_steer]) # just for convenience, does not change types of variable msgs.
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.p.s, state.p.x_tran, state.p.e_psi])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.p.s      = q[0]
        state.p.x_tran = q[1]
        state.p.e_psi    = q[2]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.p.s      = q[0]
            state.p.x_tran = q[1]
            state.p.e_psi    = q[2]
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            prediction.s      = array.array('d', q[:,0])
            prediction.x_tran = array.array('d', q[:,1])
            prediction.e_psi    = array.array('d', q[:,2])
        if u is not None:
            prediction.u_a = array.array('d', u[:,0])
            prediction.u_steer = array.array('d', u[:,1])

    def modelconstraint(self, road_curvature, sym_state, sym_control):
        '''
        wraps up model initialization
        Alternative to precompute_model.
        (1) because our one is LTV model.
        (2) Casadi nlp cannot handle ca.Function type constraints
        require the following fields to be initialized:
        self.sym_q:  ca.MX with elements of state vector q
        self.sym_u:  ca.MX with elements of control vector u
        self.sym_dq: ca.MX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''
        # state and state derivative functions
        self.sym_q  = sym_state
        self.sym_u  = sym_control

        self.sym_s = self.sym_q[0]
        self.sym_y = self.sym_q[1]
        self.sym_e_psi = self.sym_q[2]

        self.sym_u_vx = self.sym_u[0]
        self.sym_u_curvature = self.sym_u[1]

        # time derivatives
        self.sym_ds = self.sym_u_vx * ca.cos(self.sym_e_psi)
        self.sym_dy = self.sym_u_vx * ca.sin(self.sym_e_psi)
        # self.sym_ds = self.sym_u_vx
        # self.sym_dy = self.sym_u_vx*self.sym_e_psi
        self.sym_de_psi =  (self.sym_u_curvature - road_curvature) * self.sym_u_vx

        self.sym_dq = ca.vertcat(self.sym_ds, self.sym_dy, self.sym_de_psi)
        # self.sym_dq = ca.vertcat(self.sym_ds, self.sym_dsdot, self.sym_dy, self.sym_de_psi)

        sym_q_kp1 = self.sym_q + self.dt*self.sym_dq

        return sym_q_kp1

class LaneKeepingModel(CasadiArterialRoadDrivingModel):
    '''
    1-D longitudinal model for lane keeping and stop mode.
    '''
    def __init__(self, model_config): #, sym_state, sym_control):
        super().__init__(model_config)
        self.dt = 0.1
        self.M = self.model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

        self.L_f = 1.4
        self.L_r = 1.4
        self.m = self.model_config.mass
        self.g = self.model_config.gravity

        self.n_q = 1 # num of state : [station(s)]
        self.n_u = 1 # num of control : [longitudinal speed[vx]]

        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.p.s])
        u = np.array([state.u.u_a]) # just for convenience, does not change types of variable msgs.
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.p.s])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.p.s      = q[0]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.p.s      = q[0]
        if u is not None:
            state.u.u_a = u[0]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            prediction.s      = array.array('d', q[:,0])
        if u is not None:
            prediction.u_a = array.array('d', u[:,0])

    def modelconstraint(self, road_curvature, sym_state, sym_control):
        '''
        wraps up model initialization
        Alternative to precompute_model.
        (1) because our one is LTV model.
        (2) Casadi nlp cannot handle ca.Function type constraints
        require the following fields to be initialized:
        self.sym_q:  ca.MX with elements of state vector q
        self.sym_u:  ca.MX with elements of control vector u
        self.sym_dq: ca.MX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''
        # state and state derivative functions
        self.sym_q  = sym_state
        self.sym_u  = sym_control

        self.sym_s = self.sym_q[0]

        self.sym_u_vx = self.sym_u[0]

        # time derivatives
        self.sym_ds = self.sym_u_vx
        self.sym_dq = ca.vertcat(self.sym_ds)
        sym_q_kp1 = self.sym_q + self.dt*self.sym_dq

        return sym_q_kp1
