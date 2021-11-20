import random as rd
import numpy as np


def get_w(b, cov): #creates random gaussian w with mean b and covariance cov
    return np.array([rd.gauss(b[0], cov[0][0]), rd.gauss(b[1], cov[1][1])]).reshape(2, 1)


def calc_system(a_mat, b_mat, s, a, w): #calculates the next step
    return a_mat @ s + b_mat * a + w


def get_s_zero(): #calculates first (random) step s0
    return np.array([rd.gauss(0, 1), rd.gauss(0, 1)]).reshape(2, 1)


def calc_a(k_mat, s, k, r, controller_type): #calculates a
    """
    calculates input with linear nonoptimal controller, see calc_a_opt for optimal controller
    :param k_mat:
    :param s:
    :param k:
    :param r:
    :param controller_type:
    :return: returns next input for system
    """
    if controller_type is 'none':
        return - k_mat @ s + k
    elif controller_type is 'p':
        return k_mat @ (r - s) + k
    elif controller_type is 'p_zero':
        return - k_mat @ s + k
    elif controller_type is 'opt':
        return k_mat @ s + k
    else:
        return 0


def calc_reward(time, t, s, r, r_mat, a, h_mat): #reward function, t is time steps, time is time Horizon
    """
    :param time:
    :param t:
    :param s:
    :param r:
    :param r_mat:
    :param a:
    :param h_mat:
    :return: reward for timestep with LQR
    """
    rew = - (s - r).reshape(1, 2) @ r_mat @ (s - r)
    if t <= time:
        rew -= a * h_mat * a
    return rew


def get_r_mat(t): #returns the right r matix for the right time step
    """
    :param t:
    :return: matrix for punishing deviation
    """
    if t == 14 or t == 40:
        return np.array([[10**5, 0], [0, 0.1]])
    else:
        return np.array([[0.01, 0], [0, 0.1]])


def get_r(t): # returns the right r vector for the right time step
    """
    :param t:
    :return: trajectory over time
    """
    if t < 15:
        return np.array([10, 0]).reshape(2, 1)
    else:
        return np.array([20, 0]).reshape(2, 1)


def calc_q_mat(reward, a_mat, s, b_mat, a, v_mat_prev, cov): #calculates the Q funcion (bellmans recipe part 2)
    """
    :param reward:
    :param a_mat:
    :param s:
    :param b_mat:
    :param a:
    :param v_mat_prev:
    :param cov:
    :return:
    """
    return reward - (a_mat * s + b_mat * a).reshape(1, 2) * v_mat_prev *(a_mat * s + b_mat * a) + np.trace(v_mat_prev * cov)


def get_k_mat(h_mat, b_mat, v_mat_prev, a_mat): # calculates k matrix
    """
    :param h_mat:
    :param b_mat:
    :param v_mat_prev:
    :param a_mat:
    :return:
    """
    inv_part = (h_mat + b_mat.T @ v_mat_prev @ b_mat)
    return - inv_part**-1 * b_mat.T @ v_mat_prev @ a_mat


def get_k(h_mat, b_mat, v_mat_prev, v_prev, b): # calculates k
    """
    :param h_mat:
    :param b_mat:
    :param v_mat_prev:
    :param v_prev:
    :param b:
    :return:
    """
    inv_part = (h_mat + b_mat.T  @ v_mat_prev @ b_mat)
    return - inv_part**-1 * b_mat.T @ (v_mat_prev @ b - v_prev)


def get_m(h_mat, b_mat, v_mat_prev, a_mat):
    """
    :param h_mat:
    :param b_mat:
    :param v_mat_prev:
    :param a_mat:
    :return:
    """
    inv_part = (h_mat + b_mat.T @ v_mat_prev @ b_mat)
    return b_mat * inv_part**-1 * b_mat.T @ v_mat_prev @ a_mat


def get_v_mat(r_mat, a_mat, m_mat, v_mat_prev, t, time):
    """
    :param r_mat:
    :param a_mat:
    :param m_mat:
    :param v_mat_prev:
    :param t: current time
    :param time: time steps
    :return: next V iteration
    """
    if t < time:
        return r_mat + (a_mat + m_mat).T @ v_mat_prev @ a_mat
    else:
        return r_mat


def get_v(r_mat, r, a_mat, m_mat, v_prev, v_mat_prev, b, t, time):
    """
    :param r_mat:
    :param r:
    :param a_mat:
    :param m_mat:
    :param v_prev:
    :param v_mat_prev:
    :param b:
    :param t:
    :param time:
    :return: next v iteration
    """
    if t < time:
        return r_mat @ r + (a_mat + m_mat).T @ (v_prev - v_mat_prev @ b)
    else:
        return r_mat @ r




#does tha thing for x steps
def run_system(time, controller_type):
    #init first values
    s = get_s_zero()
    a_mat = np.array([[1, 0.1], [0, 1]])
    b_mat = np.array([0, 0.1]).reshape(2, 1)
    b = np.array([5, 0]).reshape(2, 1)
    cov = np.array([[0.01, 0], [0, 0.01]])
    k_mat = np.array([5, 0.3])
    k = 0.3
    h_mat = 1
    erg = []
    rew = []
    v_mat_prev = []
    m_mat = []
    v_prev = []

    for t in range(time, 0, -1): # for each time step:

        # get next values
        r_mat = get_r_mat(t)
        r = get_r(t)

        #First iteration
        if t == time:
            s = get_s_zero()
            if controller_type is 'opt':
                v_mat_prev = get_v_mat(r_mat, a_mat, m_mat, v_mat_prev, t, time)
                v_prev = get_v(r_mat, r, a_mat, m_mat, v_prev, v_mat_prev, b, t, time)

            erg.append(s)
            continue

        #Next Iteration
        if controller_type is 'opt':
            k_mat = get_k_mat(h_mat, b_mat, v_mat_prev, a_mat)
            k = get_k(h_mat, b_mat, v_mat_prev, v_prev, b)
            m_mat = get_m(h_mat, b_mat, v_mat_prev, a_mat)
            v_prev = get_v(r_mat, r, a_mat, m_mat, v_prev, v_mat_prev, b, t, time)
            v_mat_prev = get_v_mat(r_mat, a_mat, m_mat, v_mat_prev, t, time)

        a = calc_a(k_mat, s, k, r, controller_type)
        w = get_w(b, cov)
        s = calc_system(a_mat, b_mat, s, a, w)
        erg.append(s)
        if controller_type is 'none':
            reward = calc_reward(time, t, s, r, r_mat, a, h_mat)
            rew.append(reward)
    return erg, rew


