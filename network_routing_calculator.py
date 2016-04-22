# -*- coding: utf-8 -*-
"""
Given network configuration script determines the routing parameters for the given
Apos and Alen and is used to determine valid subsampling values
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_2d_routing_params(c_spacing, size_arr, a_arr, g_pos, g_len):

    # Maximum shifts at each level
    level_m = np.zeros_like(a_arr)
    m_cumm = 0
    for a_idx, a in enumerate(a_arr):
        m = (a - 1) / 2.0
        m_cumm += m
        level_m[a_idx] = m_cumm

    level_m = level_m * c_spacing

    # Level specific calculations
    theta_curr = np.array([0, 0])
    sf_arr = []

    size_max = size_arr[0][0]

    for i in np.arange(len(size_arr)-1, stop=0, step=-1):

        size_curr = size_arr[i][0]
        size_prev = size_arr[i-1][0]

        mu_i_arr_x = np.zeros(shape=size_arr[i])
        mu_i_arr_y = np.zeros(shape=size_arr[i])

        # Subsampling factor
        sf = (min(max(g_len*size_max, size_curr), size_prev) - 1) / (size_curr - 1.0)

        # Relative shift
        theta_prev = [0, 0]
        for d_idx in np.arange(len(g_pos)):
            if np.abs(g_pos[d_idx]) >= level_m[i-1]:
                theta_prev[d_idx] = g_pos[d_idx] - np.sign(g_pos[d_idx]) * level_m[i - 1]

        # shift
        s = theta_prev - theta_curr

        # focus of each column
        column_pos_x = np.arange(size_arr[i][0]) * c_spacing - size_arr[i][0] / 2 * c_spacing
        column_pos_y = np.arange(size_arr[i][0]) * c_spacing - size_arr[i][0] / 2 * c_spacing

        for x_idx, x in enumerate(column_pos_x):
            for y_idx, y in enumerate(column_pos_y):
                mu_i_arr_x[x_idx, y_idx] = sf*x + s[0]
                mu_i_arr_y[x_idx, y_idx] = sf*y + s[1]

        theta_curr = theta_prev
        sf_arr.append(sf)

        print("Level %d:" % (i + 1))
        print("sf=%0.2f, theta prev=%s, shift=%s" % (sf, theta_prev, s))
        print("mu_i as pos:focus")
        for x_idx, x in enumerate(column_pos_x):
            for y_idx, y in enumerate(column_pos_y):
                print ("(%0.2f,%0.2f):(%0.2f,%0.2f)"
                       % (x, y, mu_i_arr_x[x_idx, y_idx], mu_i_arr_y[x_idx, y_idx]))

    return sf_arr


def calculate_1d_routing_params(c_spacing, size_arr, a_arr, g_pos, g_len):
    """

    :param c_spacing:
    :param size_arr: lowest level to highest level
    :param a_arr: number of afferent connections at each level
    :param g_pos: Apos
    :param g_len: Alen
    :return:
    """

    # Maximum shifts at each level
    level_m = np.zeros_like(a_arr)
    m_cumm = 0
    for a_idx, a in enumerate(a_arr):
        m = (a - 1) / 2.0
        m_cumm += m
        level_m[a_idx] = m_cumm

    level_m = level_m * c_spacing

    # Level specific calculations
    theta_curr = 0
    sf_arr = []

    size_max = size_arr[0]

    for i in np.arange(len(size_arr)-1, stop=0, step=-1):
        sf = (min(max(g_len*size_max, size_arr[i]), size_arr[i - 1]) - 1) / (size_arr[i] - 1.0)

        if np.abs(g_pos) < level_m[i-1]:
            theta_prev = 0
        else:
            theta_prev = g_pos - np.sign(g_pos) * level_m[i - 1]

        s = theta_prev - theta_curr

        column_pos = np.arange(size_arr[i]) * c_spacing - size_arr[i] / 2 * c_spacing

        mu_i_arr = sf * column_pos + s

        theta_curr = theta_prev
        sf_arr.append(sf)

        print("Level %d:" % (i + 1))
        print("m=%0.2f, sf=%0.2f, theta prev=%0.2f, shift=%0.2f" % (level_m[i], sf, theta_prev, s))
        print "pos:focus " + str(["%0.2f:%0.2f" % (pos, mu_i_arr[p_idx])
                                  for p_idx, pos in enumerate(column_pos)])

    return sf_arr


if __name__ == '__main__':
    plt.ion()

    # -------------------------------------------------------------------------------
    # 1D routing Example
    # -------------------------------------------------------------------------------
    between_column_dist = 0.25

    level_sizes = np.array([9, 7, 3])  # Lowest Level to highest level
    level_a = np.array([1, 3, 5])      # Number of afferent connections at each level
    # At lowest level a = 1, this evaluates to m=0 (no shifting at lowest level)

    max_size = np.float(level_sizes[0])

    valid_sf = set()

    # All valid positions and sizes
    Alen_arr = np.arange(level_sizes[-1], level_sizes[0] + 1, step=2)
    Apos_arr = (level_sizes[0] - Alen_arr) / 2 * between_column_dist

    for r_idx in np.arange(len(Alen_arr)):

        Apos = Apos_arr[r_idx]
        Alen = Alen_arr[r_idx] / max_size
        print("Apos=%0.2f, Alen=%0.2f ------------------------------------" % (Apos, Alen))

        sfs = calculate_1d_routing_params(
            c_spacing=between_column_dist,
            size_arr=level_sizes,
            a_arr=level_a,
            g_pos=Apos,
            g_len=Alen
        )

        # print sfs
        for item in sfs:
            valid_sf.add(item)

    print("-"*80)
    print ("Valid Sf values %s" % valid_sf)

    # # -------------------------------------------------------------------------------
    # # 2D routing Example
    # # -------------------------------------------------------------------------------
    # between_column_dist = 0.25
    #
    # level_sizes = np.array([(7, 7), (3, 3)])  # Lowest Level to highest level
    # level_a = np.array([1, 5])      # Number of afferent connections at each level
    #
    # # If 2 dimensional
    # max_size = level_sizes[0][0]
    #
    # valid_sf = set()
    #
    # Apos = (-0.25, 0.25)
    # Alen = 3/7.
    #
    # print("Apos=%s, Alen=%0.2f ------------------------------------" % (Apos, Alen))
    # sfs = calculate_2d_routing_params(
    #     c_spacing=between_column_dist,
    #     size_arr=level_sizes,
    #     a_arr=level_a,
    #     g_pos=Apos,
    #     g_len=Alen
    # )
