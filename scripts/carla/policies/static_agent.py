import carla
import os
import sys
import numpy as np
import time

class StaticAgent(object):
    """ A path following agent with collision avoidance constraints over a short horizon. """

    def __init__(self, vehicle, goal_location):
        self.vehicle = vehicle

        self.static_control = carla.VehicleControl()
        self.static_control.hand_brake = False
        self.static_control.manual_gear_shift = False
        self.static_control.throttle =  0.
        self.static_control.brake    = -1.
        self.static_control.steer    =  0.

        # goal_location ignored because this agent doesn't move.

    def done(self):
        return True

    def run_step(self, **kwargs):
        vehicle_tf    = self.vehicle.get_transform()
        vehicle_vel   = self.vehicle.get_velocity()

        # Get the vehicle's current pose + speed in a RH coordinate system.
        x, y = vehicle_tf.location.x, -vehicle_tf.location.y
        psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))
        speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)

        z0 = np.array([x, y, psi, speed]) # current kinematic state
        u0 = np.array([0., 0.])           # acceleration, steering angle setpoint for low-level control

        return self.static_control, z0, u0