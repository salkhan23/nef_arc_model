# -*- coding: utf-8 -*-
"""
2D ARC model with two levels.

Network Settings

            Columns      RF size (a)
L2 (top)  :     3x3,              5,
L1(bottom):     7x7,              -
"""

import numpy as np
import matplotlib.pyplot as plt
import nengo

import cortical_column as cc
import network_routing_calculator as nrc

reload(cc)
reload(nrc)


def get_unraveled_2dim_positions(size, c_spacing):
    """
     returns a 1D mapping of the column positions for a level in the same order as np.unravel
     starting from the top level and moving row first

    :param size: must be 2d
    :param c_spacing:
    :return:
    """
    pos_arr = []

    pos_x = (np.arange(size[0]) - int(size[0])/2) * c_spacing
    pos_y = (np.arange(size[1]) - int(size[1])/2) * c_spacing
    pos_y = pos_y[::-1]  # unravel moves from left to right

    for y in pos_y:
        for x in pos_x:
            pos_arr.append((x, y))

    return np.array(pos_arr)


if __name__ == '__main__':
    plt.ion()

    # Common Settings
    tau_ref = 0.002
    tau_rc = 0.02
    tau_psc = 0.1  # post synaptic time constant

    time_stop = 5

    # Network wide settings
    level2_a = 5  # This is a radius, in total there are a^2 number of connections

    level2_size = np.array([3, 3])
    level1_size = np.array([7, 7])

    valid_sampling_factors = [1, 2, 3]
    between_column_dist = 0.25
    max_size = np.float(level1_size[0])  # Number of columns in lowest level

    # Global Feedback Signals (from Pulvinar)
    A_len = 3 / 7.0
    A_pos = np.array([0.5, 0.5])
    A_theta = np.array([0, 0])  # At highest level = 0, object representation is assumed centered

    # Printed out the expected routing parameters
    print("Routing Details Apos=%s, Alen=%0.2f  ---------------------------------"
          % (A_pos, A_len))

    nrc.calculate_2d_routing_params(
        c_spacing=between_column_dist,
        size_arr=np.array([level1_size, level2_size]),
        a_arr=np.array([1, level2_a]),
        g_len=A_len,
        g_pos=A_pos,
    )
    print('-' * 80)

    # Build the  Network
    model = nengo.Network(label='2D ARC with 2 Level network')

    # # -----------------------------------------------------------------------------------
    # # Model Inputs
    # # -----------------------------------------------------------------------------------
    with model:
        # Feed back inputs
        stim_g_pos = nengo.Node(A_pos)
        stim_g_len = nengo.Node(A_len)
        stim_g_theta = nengo.Node(A_theta)

        # Feed forward inputs
        # Input to the model is a Z in the top right corner and a diamond at the bottom left
        level1_column_inputs = np.array([
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ])

        level1_column_inputs = level1_column_inputs.ravel()
        level1_column_pos_arr = get_unraveled_2dim_positions(level1_size, between_column_dist)

        stim_inputs = nengo.Node(level1_column_inputs)

    # -----------------------------------------------------------------------------------
    # Level 1 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 1")
    # position mapping of layer one is the same as the input
    level1_m = 0  # No shift at lowest level

    with model:
        level1 = []
        level1_pop_probes = []
        for i_idx in np.arange(len(level1_column_pos_arr)):
            column = nengo.Ensemble(
                100,  # Population size
                1,    # Dimensions
                max_rates=nengo.dists.Uniform(100, 200),
                neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
                label='Level1, column position %s' % (level1_column_pos_arr[i_idx]),
            )

            nengo.Connection(stim_inputs[i_idx], column, synapse=tau_psc)
            level1_pop_probes.append(nengo.Probe(column, synapse=0.1))
            level1.append(column)

    # -----------------------------------------------------------------------------------
    # Level 2 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 2")

    level2_column_pos_arr = []
    level2_column_pos_arr = get_unraveled_2dim_positions(level2_size, between_column_dist)
    level2_m = (level2_a - 1) / 2.0 * between_column_dist + level1_m

    with model:
        # Common to the Level ---------------------------------
        # Layer 5 position - locally represents global position and theta at current level
        level2_l5_pos = nengo.Ensemble(
            900,  # population size
            4,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 position, theta_curr',
            radius=np.sqrt(4)
        )

        # Layer 5 size - locally represent global size
        level2_l5_size = nengo.Ensemble(
            300,  # population size
            1,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 size',
        )

        # Columns in Level 2
        offsets = get_unraveled_2dim_positions((level2_a, level2_a), between_column_dist)

        level2_columns = []
        for c_idx, c_pos in enumerate(level2_column_pos_arr):
            prev_level_column_pos = c_pos + offsets

            # Corresponding indices in lower level array
            prev_level_column_idxs = \
                [np.where(np.all(level1_column_pos_arr == pos, axis=1))[0]
                 for pos in prev_level_column_pos]

            prev_level_column_idxs = np.array(prev_level_column_idxs)
            prev_level_column_idxs = prev_level_column_idxs.ravel()

            ff_in_prev_level = [level1[int(idx)] for idx in prev_level_column_idxs]

            column = cc.TwoDimCorticalColumn(
                t_ref=tau_ref, t_rc=tau_rc, t_psc=tau_psc,
                c_spacing=between_column_dist,
                m=level2_m,
                m_prev=level1_m,
                i=c_pos,
                prev_c_positions=prev_level_column_pos,
                size_max=max_size,
                size_curr_l=level2_size,
                size_prev_l=level1_size,
                prev_c_out_nodes=ff_in_prev_level,
                l5_pos_node=level2_l5_pos,
                l5_size_node=level2_l5_size,
                valid_sf=valid_sampling_factors
            )
            level2_columns.append(column)

    # -----------------------------------------------------------------------------------
    # Feedback Connections
    # -----------------------------------------------------------------------------------
    with model:
        nengo.Connection(stim_g_pos[0], level2_l5_pos[0], synapse=tau_psc)
        nengo.Connection(stim_g_pos[1], level2_l5_pos[1], synapse=tau_psc)

        nengo.Connection(stim_g_theta[0], level2_l5_pos[2], synapse=tau_psc)
        nengo.Connection(stim_g_theta[1], level2_l5_pos[3], synapse=tau_psc)

        nengo.Connection(stim_g_len, level2_l5_size, synapse=tau_psc)

    # Run the model --------------------------------------------------------------------
    sim = nengo.Simulator(model)
    sim.run(time_stop)
    t = sim.trange()

    # Plot average outputs of each column at each level vs position --------------------
    f, ax_arr = plt.subplots(1, 3)

    # Inputs
    lev1_x_min = -level1_size[0]/2 + 1
    lev1_x_max = level1_size[0]/2
    lev1_y_min = -level1_size[1]/2 + 1
    lev1_y_max = level1_size[1]/2

    ax_arr[0].matshow(
        level1_column_inputs.reshape(level1_size[0], level1_size[0]),
        extent=[lev1_x_min, lev1_x_max, lev1_y_min, lev1_y_max],
        cmap=plt.cm.coolwarm,
        vmin=0, vmax=1,
    )
    ax_arr[0].set_title("Inputs to Level 1")
    ax_arr[0].set_ylabel("Y")
    ax_arr[0].set_xlabel("X")

    # level 1
    avg_level1_out = []
    for probe in level1_pop_probes:
        avg_level1_out.append(np.mean(sim.data[probe]))

    avg_level1_out = np.array(avg_level1_out)
    im = ax_arr[1].matshow(
        avg_level1_out.reshape(level1_size[0], level1_size[0]),
        extent=[lev1_x_min, lev1_x_max, lev1_y_min, lev1_y_max],
        cmap=plt.cm.coolwarm,
        vmin=0, vmax=1,
    )
    ax_arr[1].set_title("Average output level 1")
    ax_arr[1].set_xlabel("X")

    # Level 2
    avg_level2_out = []
    for column in level2_columns:
        avg_level2_out.append(np.mean(sim.data[column.l2_3_output_p]))

    avg_level2_out = np.array(avg_level2_out)
    ax_arr[2].matshow(
        avg_level2_out.reshape(level2_size[0], level2_size[0]),
        extent=[-level2_size[0]/2 + 1, level2_size[0]/2,
                -level2_size[1]/2 + 1, level2_size[1]/2],
        cmap=plt.cm.coolwarm,
        vmin=0, vmax=1,
    )

    ax_arr[2].set_title("Average output level 2")
    ax_arr[2].set_xlabel("X")

    cbar_ax = f.add_axes([0.15, 0.15, 0.7, 0.02])
    f.colorbar(im, cax=cbar_ax, orientation='horizontal')
