# -*- coding: utf-8 -*-
"""
1D model with two levels.

Network Settings

            Columns      RF size (a)
L2        :      3,               3,
L1(bottom):      5,               -
"""

import numpy as np
import matplotlib.pyplot as plt
import nengo

import cortical_column as cc
import network_routing_calculator as nrc

reload(cc)
reload(nrc)

if __name__ == '__main__':
    plt.ion()

    # Common Settings
    tau_ref = 0.002
    tau_rc = 0.02
    tau_psc = 0.1  # post synaptic time constant

    time_stop = 5

    # Network wide settings
    level2_a = 3  # Number of afferent connections into each L2 column

    level1_size = 5
    level2_size = 3

    valid_sampling_factors = [1, 1.5, 2]
    between_column_dist = 0.25
    max_size = 5.0  # Number of columns in lowest level

    # Global Feedback Signals (from Pulvinar)
    A_len = 5 / 5.0
    A_pos = 0
    A_theta = 0  # At highest level = 0, object representation should always be centered

    # Printed out the expected routing parameters
    print("Routing Details Apos=%0.2f, Alen=%0.2f" % (A_pos, A_len))
    nrc.calculate_1d_routing_params(
        c_spacing=between_column_dist,
        size_arr=np.array([level1_size, level2_size]),
        a_arr=np.array([0, level2_a]),
        g_len=A_len,
        g_pos=A_pos,
    )
    print('-' * 80)

    # Build the  Network
    model = nengo.Network(label='1D ARC 2 levels')

    # -----------------------------------------------------------------------------------
    # Model Inputs
    # -----------------------------------------------------------------------------------
    with model:
        # Feed back inputs
        stim_g_pos = nengo.Node(A_pos)
        stim_g_len = nengo.Node(A_len)
        stim_g_theta = nengo.Node(A_theta)

        # Feed forward inputs
        level1_column_inputs = [0, 0, 1, 1, 1]  # Object in the right end of the visual field
        stim_inputs = nengo.Node(level1_column_inputs)

    # -----------------------------------------------------------------------------------
    # Level 1 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 1")
    level1_column_pos_arr = [-0.5, -0.25, 0, 0.25, 0.5]
    level1_m = 0  # No shifting at lowest level

    with model:
        level1 = []
        level1_pop_probes = []
        for i_idx in np.arange(len(level1_column_pos_arr)):
            column = nengo.Ensemble(
                100,  # Population size
                1,  # Dimensions
                max_rates=nengo.dists.Uniform(100, 200),
                neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
                label='Level1, column position %0.2f' % (level1_column_pos_arr[i_idx]),
            )

            nengo.Connection(stim_inputs[i_idx], column, synapse=tau_psc)
            level1_pop_probes.append(nengo.Probe(column, synapse=0.1))
            level1.append(column)

    # -----------------------------------------------------------------------------------
    # Level 2 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 2")
    level2_column_pos_arr = [-0.25, 0, 0.25]
    level2_m = 1 * between_column_dist

    with model:
        # Common to the Level
        # Layer 5 position - locally represents global position and theta at current level
        level2_l5_pos = nengo.Ensemble(
            300,  # population size
            2,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 position, theta_curr',
            radius=np.sqrt(2)
        )

        # Layer 5 size - locally represent global size
        level2_l5_size = nengo.Ensemble(
            300,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 size',
        )

        # Columns in Level 2
        max_shift_columns = level2_m / between_column_dist
        idxs = np.arange(-max_shift_columns, max_shift_columns + 1)  # include end point

        level2_columns = []
        for c_idx, c_pos in enumerate(level2_column_pos_arr):
            prev_level_column_pos = c_pos + idxs * between_column_dist

            prev_level_column_idxs = max_shift_columns + c_idx + idxs
            ff_in_prev_level = [level1[int(idx)] for idx in prev_level_column_idxs]

            column = cc.CorticalColumn(
                t_ref=tau_ref, t_rc=tau_rc, t_psc=tau_psc,
                c_spacing=between_column_dist,
                m=level2_m,
                m_prev=level1_m,
                i=level2_column_pos_arr[c_idx],
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
        nengo.Connection(stim_g_pos, level2_l5_pos[0], synapse=tau_psc)
        nengo.Connection(stim_g_theta, level2_l5_pos[1], synapse=tau_psc)
        nengo.Connection(stim_g_len, level2_l5_size, synapse=tau_psc)

    # Run the model --------------------------------------------------------------------
    sim = nengo.Simulator(model)
    sim.run(time_stop)
    t = sim.trange()

    # Plot the outputs of each column --------------------------------------------------
    f, ax_arr = plt.subplots(2, 1, sharex=True)
    fb, ax_arr_b = plt.subplots(2, 1, sharex=True)

    # level 1 plots
    for p_idx, probe in enumerate(level1_pop_probes):
        ax_arr_b[1].plot(t, sim.data[probe],
                         label='L1 column %0.2f' % level1_column_pos_arr[p_idx])
        ax_arr[1].scatter(
            level1_column_pos_arr[p_idx] * np.ones_like(sim.data[probe]), sim.data[probe])

    # level 2 plots
    for c_idx, column in enumerate(level2_columns):
        ax_arr_b[0].plot(t, sim.data[level2_columns[c_idx].l2_3_output_p],
                         label='L2 column %0.2f' % column.i)
        ax_arr[0].scatter(
            level2_column_pos_arr[c_idx] * np.ones_like(sim.data[column.l2_3_output_p]),
            sim.data[column.l2_3_output_p])

    for idx in np.arange(len(ax_arr)):
        level_idx = 2 - idx

        ax_arr[idx].set_ylim([-0.1, 1.1])
        ax_arr[idx].legend()
        ax_arr[idx].set_title("Level %d outputs" % level_idx)

        ax_arr_b[idx].set_ylim([-0.1, 1.1])
        ax_arr_b[idx].legend()
        ax_arr_b[idx].set_title("Level %d outputs" % level_idx)

    # Plot the output of each dendrites for each column ---------------------------------
    # Level 2
    f2, ax_arr = plt.subplots(int(level2_size))
    f2.suptitle("Level2 Dendrites output")

    for c_idx, column in enumerate(level2_columns):
        ax_arr[c_idx].set_title("L2 column at position %0.2f" % column.i)

        for l4_idx, l4 in enumerate(column.l4_arr):
            ax_arr[c_idx].plot(t, sim.data[l4.soma_p],
                               label='scaled output for L1 column at position %0.2f'
                                     % column.prev_c_positions[l4_idx])
        ax_arr[c_idx].legend()
        ax_arr[c_idx].set_ylim([-0.1, 1.1])
