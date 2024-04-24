import carla
import os
import sys
import numpy as np

class LowLevelControl:
    def __init__(self, vehicle):
        # Control setup and parameters.
        self.control_prev = carla.VehicleControl()
        self.max_steer_angle = np.radians( vehicle.get_physics_control().wheels[0].max_steer_angle )
        self.alpha         = 0.4 # low-pass filter on actuation to simulate first order delay

        # Throttle Parameters
        self.k_v  = 0.9  # P gain on velocity tracking error
        self.thr_ff_map  = np.column_stack(([  2.5,  7.5,  12.5,  17.5],        # speed (m/s) -> steady state throttle
                                            [0.325, 0.45, 0.525, 0.625]))

        # Brake Parameters
        self.brake_accel_thresh = -2.0 # m/s^2, value below which the brake is activated
        self.brake_decel_map  = np.column_stack(([ 1.6,  3.9, 6.8,  7.1, 7.9],  # deceleration (m/s^2) -> steady state throttle (at 12 m/s^2)
                                                 [  0., 0.25, 0.5, 0.75, 1.0]))

    def update(self, v_curr, a_des, v_des, df_des):
        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False

        if a_des > self.brake_accel_thresh:
            control.throttle = self.k_v * (v_des - v_curr) + np.interp(v_des, self.thr_ff_map[:,0], self.thr_ff_map[:,1])
        else:
            control.brake    = np.interp( -a_des, self.brake_decel_map[:,0], self.brake_decel_map[:,1])

        # Simulated actuation delay, also used to avoid high frequency control inputs.
        if control.throttle > 0.0:
            control.throttle = self.alpha * control.throttle + (1. - self.alpha) * self.control_prev.throttle

        elif control.brake > 0.0:
            control.brake    = self.alpha * control.brake    + (1. - self.alpha) * self.control_prev.brake

        # Steering control.  Flipped sign due to Carla LHS convention.

        control.steer    = -df_des / self.max_steer_angle
        control.steer    = self.alpha * control.steer    + (1. - self.alpha) * self.control_prev.steer

        # Clip Carla control to limits.
        control.throttle = np.clip(control.throttle, 0.0, 1.0)
        control.brake    = np.clip(control.brake, 0.0, 1.0)
        control.steer    = np.clip(control.steer, -1.0, 1.0)

        self.control_prev = control

        return control
