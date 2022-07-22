#!/usr/bin/env python

import numpy as np
import math as m

LOOP_DIST = 1036
def frenet2global(s_cur,ey_cur,mat):
    #mat : self.mat = scipy.io.loadmat('Road_test_Both.mat')
    s = mat['road_s']
    x = mat['road_x']
    y = mat['road_y']
    theta = mat['road_yaw']
    gap = (s-s_cur)**2
    idx_min = np.argmin(gap)
    x_cur = x[idx_min][0]
    y_cur = y[idx_min][0]
    theta_cur = theta[idx_min][0]
    x_cur = x_cur - ey_cur*m.sin(theta_cur)
    y_cur = y_cur + ey_cur*m.cos(theta_cur)
    return x_cur, y_cur, idx_min

def frenet2global4mpc(s_cur,ey_cur,mat):
    #mat : self.mat = scipy.io.loadmat('Road_test_Both.mat')
    s = mat['road_plan_s']
    y = mat['road_plan_y']
    gap = (s-s_cur)**2
    idx_min = np.argmin(gap)
    refy_cur4map = y[idx_min]
    ey4map = ey_cur + refy_cur4map
    return ey4map

def global2frenet(mat, x_cur, y_cur, psi_cur):
    s = mat['road_s']
    x = mat['road_x']
    y = mat['road_y']
    K = mat['road_K']
    psi = mat['road_yaw']

    norm_array = (x-x_cur)**2+(y-y_cur)**2
    idx_min = np.argmin(norm_array)

    s_cur = s.item(idx_min)

    # unsigned ey
    e_y_cur = np.sqrt(norm_array.item(idx_min))

    # ey sign
    delta_x = x.item(idx_min) - x_cur
    delta_y = y.item(idx_min) - y_cur
    delta_vec = np.array([[delta_x], [delta_y]])
    # unit_vec = np.array([[np.cos(psi.item(idx_min))], [np.sin(psi.item(idx_min))]])
    # R: Rotation matrix, C:
    R = np.array([[np.cos(psi.item(idx_min)), np.sin(psi.item(idx_min))], [- np.sin(psi.item(idx_min)), np.cos(psi.item(idx_min))]])
    C = np.array([0,1]).reshape((1,-1))
    ey_dir = - np.sign(C @ R @ (delta_vec)) # - (delta_vec.T @ unit_vec) * unit_vec))

    # signed ey
    e_y_cur = e_y_cur * ey_dir

    e_psi_cur = psi_cur - psi.item(idx_min)

    # If psi_cur = pi, psi.item(idx_min) = -pi then e_psi_cur = 2pi although it represent the same angle
    # we adjust it
    if e_psi_cur>= np.pi:
        e_psi_cur -= 2*np.pi
    elif e_psi_cur <= -np.pi:
        e_psi_cur += 2*np.pi
    return s_cur, e_y_cur, e_psi_cur, idx_min

def curvatures4LKouput(mat, s_sol):
    s = mat['road_s']
    K = mat['road_K']
    # print('shape : {}'.format(s_sol.shape))
    K_n = s_sol.shape[0]

    K_array = np.ones((K_n,))

    for ind in range(K_n):
        s_sol_ind = s_sol.item(ind)
        s_err_array = np.abs(s - s_sol_ind)
        idx_min = np.argmin(s_err_array)

        K_array[ind] = K.item(idx_min)

    # K_array.squeeze()
    K_array.reshape((1,-1))

    # print(K_array.shape)

    return K_array

def curvatures4planner(mat, idx_min, N_MPC, Ts, s_cur, vx_cur, ax_cur):
    s = mat['road_s']
    K = mat['road_K']
    K_array = np.ones((1,N_MPC))

    ## s to K array
    mock_t_array = np.arange(N_MPC) * Ts
    mock_s_array = s_cur + vx_cur * mock_t_array + 0.5 * ax_cur * mock_t_array ** 2

    for ind in range(N_MPC):
        mock_s = mock_s_array.item(ind)
        if mock_s >=LOOP_DIST:
            mock_s -= LOOP_DIST
        s_err_array = np.abs(s - mock_s)
        idx_min = np.argmin(s_err_array)

        K_array[0, ind] = K.item(idx_min)

    K_array.reshape((1,-1))

    # print(K_array.shape)

    return K_array

def speed4planner(mat, idx_min, N_MPC, Ts, s_cur, vx_cur, ax_cur):
    s = mat['road_s']
    Vx = mat['road_Vx']
    Vx_array = np.ones((1,N_MPC))

    ## s to K array
    mock_t_array = np.arange(N_MPC) * Ts
    mock_s_array = s_cur + vx_cur * mock_t_array + 0.5 * ax_cur * mock_t_array ** 2

    for ind in range(N_MPC):
        mock_s = mock_s_array.item(ind)
        if mock_s >=LOOP_DIST:
            mock_s -= LOOP_DIST
        s_err_array = np.abs(s - mock_s)
        idx_min = np.argmin(s_err_array)

        Vx_array[0, ind] = Vx.item(idx_min)

    Vx_array.reshape((1,-1))

    return Vx_array

def curvatures4mpc(mat, idx_min, N_MPC, Ts, s_cur, vx_cur, ax_cur):
    s = mat['road_plan_s']
    K = mat['road_plan_K']
    # s = mat['road_s']
    # K = mat['road_K']
    K_array = np.ones((1,N_MPC))

    ## s to K array
    mock_t_array = np.arange(N_MPC) * Ts
    mock_s_array = s_cur + vx_cur * mock_t_array + 0.5 * ax_cur * mock_t_array ** 2

    for ind in range(N_MPC):
        mock_s = mock_s_array.item(ind)
        if mock_s >=LOOP_DIST:
            mock_s -= LOOP_DIST
        s_err_array = np.abs(s - mock_s)
        idx_min = np.argmin(s_err_array)

        K_array[0, ind] = K.item(idx_min)

    K_array.reshape((1,-1))

    # print(K_array.shape)

    return K_array

def speed4mpc(mat, idx_min, N_MPC, Ts, s_cur, vx_cur, ax_cur):
    s = mat['road_plan_s'][:-2]
    Vx = mat['road_plan_vx']
    # s = mat['road_s']
    # Vx = mat['road_Vx']
    Vx_array = np.ones((1,N_MPC))

    ## s to K array
    mock_t_array = np.arange(N_MPC) * Ts
    mock_s_array = s_cur + vx_cur * mock_t_array + 0.5 * ax_cur * mock_t_array ** 2

    for ind in range(N_MPC):
        mock_s = mock_s_array.item(ind)
        if mock_s >=LOOP_DIST:
            mock_s -= LOOP_DIST
        s_err_array = np.abs(s - mock_s)
        idx_min = np.argmin(s_err_array)

        Vx_array[0, ind] = Vx.item(idx_min)

    Vx_array.reshape((1,-1))

    return Vx_array

def error4mpc(mat, idx_min, N_MPC, Ts, s_cur, vx_cur, ax_cur):
    s = mat['road_plan_s']
    y = mat['road_plan_y']
    psi = mat['road_plan_yaw']
    y_array = np.ones((1,N_MPC))
    psi_array = np.ones((1,N_MPC))

    ## s to K array
    mock_t_array = np.arange(N_MPC) * Ts
    mock_s_array = s_cur + vx_cur * mock_t_array + 0.5 * ax_cur * mock_t_array ** 2

    for ind in range(N_MPC):
        mock_s = mock_s_array.item(ind)
        if mock_s >=LOOP_DIST:
            mock_s -= LOOP_DIST
        s_err_array = np.abs(s - mock_s)
        idx_min = np.argmin(s_err_array)

        y_array[0, ind] = y.item(idx_min)
        psi_array[0,ind] = psi.item(idx_min)

    y_array.reshape((1,-1))
    psi_array.reshape((1,-1))

    return y_array, psi_array
