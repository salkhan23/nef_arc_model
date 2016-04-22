# -*- coding: utf-8 -*-
"""
1D model with three levels.

Network Settings

            Columns      RF size (a)
L3 (top)  :      3,               5,
L2        :      7,               3,
L1(bottom):      9,               -
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
    level3_a = 5  # Number of afferent connections into each L3 column
    level2_a = 3  # Number of afferent connections into each L2 column

    level3_size = 3
    level2_size = 7
    level1_size = 9

    valid_sampling_factors = [1, 4 / 3., 2, 3]
    between_column_dist = 0.25
    max_size = np.float(level1_size)  # Number of columns in lowest level

    # Global Feedback Signals (from Pulvinar)
    A_len = 3 / 9.0
    A_pos = 0.5
    A_theta = 0  # At highest level = 0, object representation is assumed centered

    # Printed out the expected routing parameters
    print("Routing Details Apos=%0.2f, Alen=%0.2f" % (A_pos, A_len))
    nrc.calculate_1d_routing_params(
        c_spacing=between_column_dist,
        size_arr=np.array([level1_size, level2_size, level3_size]),
        a_arr=np.array([1, level2_a, level3_a]),
        g_len=A_len,
        g_pos=A_pos,
    )
    print('-' * 80)

    # Build the  Network
    model = nengo.Network(label='1D ARC with 3 Level network')

    # -----------------------------------------------------------------------------------
    # Model Inputs
    # -----------------------------------------------------------------------------------
    with model:
        # Feed back inputs
        stim_g_pos = nengo.Node(A_pos)
        stim_g_len = nengo.Node(A_len)
        stim_g_theta = nengo.Node(A_theta)

        # Feed forward inputs
        level1_column_inputs = [0, 0, 0, 0, 0, 0, 1, 1, 1]  # Object at right end of space
        stim_inputs = nengo.Node(level1_column_inputs)

    # -----------------------------------------------------------------------------------
    # Level 1 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 1")
    level1_column_pos_arr = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
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
                label='Level1, column position %0.2f' % (level1_column_pos_arr[i_idx]),
            )

            nengo.Connection(stim_inputs[i_idx], column, synapse=tau_psc)
            level1_pop_probes.append(nengo.Probe(column, synapse=0.1))
            level1.append(column)

    # -----------------------------------------------------------------------------------
    # Level 2 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 2")
    level2_column_pos_arr = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
    level2_m = (level2_a - 1) / 2.0 * between_column_dist

    with model:
        # Common to the Level ---------------------------------
        # Layer 5 position - locally represents global position and theta at current level
        level2_l5_pos = nengo.Ensemble(
            300,  # population size
            2,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Level 2 Layer 5 position, theta_curr',
            radius=np.sqrt(2)
        )

        # Layer 5 size - locally represent global size
        level2_l5_size = nengo.Ensemble(
            300,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Level 2 Layer 5 size',
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
    # Level 3 of network
    # -----------------------------------------------------------------------------------
    print("Constructing Level 3")
    level3_column_pos_arr = [-0.25, 0, 0.25]
    level3_m = (level3_a - 1) / 2.0 * between_column_dist

    with model:
        # Layer 5 position - locally represents global position and theta at current level
        level3_l5_pos = nengo.Ensemble(
            300,  # population size
            2,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Level 3 Layer 5 position, theta_curr',
            radius=np.sqrt(2)
        )

        # Layer 5 size - locally represent global size
        level3_l5_size = nengo.Ensemble(
            300,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Level 3 Layer 5 size',
        )

        # Columns in level 3
        max_shift_columns = level3_m / between_column_dist
        idxs = np.arange(-max_shift_columns, max_shift_columns + 1)  # include end point

        level3_columns = []
        for c_idx, c_pos in enumerate(level3_column_pos_arr):
            prev_level_column_pos = c_pos + idxs * between_column_dist
            prev_level_column_idxs = max_shift_columns + c_idx + idxs

            ff_in_prev_level_d = \
                [level2_columns[int(idx)].l2_3_output for idx in prev_level_column_idxs]

            column = cc.CorticalColumn(
                t_ref=tau_ref, t_rc=tau_rc, t_psc=tau_psc,
                c_spacing=between_column_dist,
                m=level3_m,
                m_prev=level2_m,
                i=c_pos,
                prev_c_positions=prev_level_column_pos,
                size_max=max_size,
                size_curr_l=level3_size,
                size_prev_l=level2_size,
                prev_c_out_nodes=ff_in_prev_level_d,
                l5_pos_node=level3_l5_pos,
                l5_size_node=level3_l5_size,
                valid_sf=valid_sampling_factors
            )
            level3_columns.append(column)
    # -----------------------------------------------------------------------------------
    # Feedback Connections
    # -----------------------------------------------------------------------------------
    with model:
        nengo.Connection(stim_g_pos, level3_l5_pos[0], synapse=tau_psc)
        nengo.Connection(stim_g_theta, level3_l5_pos[1], synapse=tau_psc)
        nengo.Connection(stim_g_len, level3_l5_size, synapse=tau_psc)

        nengo.Connection(level3_l5_pos[0], level2_l5_pos[0], synapse=tau_psc)
        # Get theta_curr_l2 = theta_prev_l3
        # we use relative shift_3 = [theta_prev_l3 - theta_curr_l3] l + theta_curr_l3
        nengo.Connection(level3_l5_pos[1], level2_l5_pos[1], synapse=tau_psc)
        nengo.Connection(level3_columns[0].l6_shift, level2_l5_pos[1], synapse=tau_psc)

        # Size cannot be negative
        eval_points = np.random.uniform(0, 1, size=(1000, 1))
        nengo.Connection(level3_l5_size, level2_l5_size, synapse=tau_psc, eval_points=eval_points)

    # Run the model --------------------------------------------------------------------
    sim = nengo.Simulator(model)
    sim.run(time_stop)
    t = sim.trange()

    # Plot the outputs of each column vs its position -----------------------------------
    f, ax_arr = plt.subplots(3, 1, sharex=True)
    f2, ax_arr2 = plt.subplots(3, 1, sharex=True)

    # level 3 plots
    for c_idx, column in enumerate(level3_columns):

        ax_arr[0].scatter(
            level3_column_pos_arr[c_idx] * np.ones_like(sim.data[column.l2_3_output_p]),
            sim.data[column.l2_3_output_p])

        ax_arr2[0].plot(t, sim.data[level3_columns[c_idx].l2_3_output_p],
                        label='L3 column %0.2f' % column.i)

    # level 2 plots
    for c_idx, column in enumerate(level2_columns):

        ax_arr[1].scatter(
            level2_column_pos_arr[c_idx] * np.ones_like(sim.data[column.l2_3_output_p]),
            sim.data[column.l2_3_output_p])

        ax_arr2[1].plot(t, sim.data[level2_columns[c_idx].l2_3_output_p],
                        label='L2 column %0.2f' % column.i)

    # level 1 plots
    for p_idx, probe in enumerate(level1_pop_probes):

        ax_arr[2].scatter(
            level1_column_pos_arr[p_idx] * np.ones_like(sim.data[probe]),
            sim.data[probe])

        ax_arr2[2].plot(t, sim.data[probe], label='L1 column %0.2f' % level1_column_pos_arr[p_idx])

    for ii in np.arange(len(ax_arr)):
        ax_arr[ii].set_ylim([-0.1, 1.1])
        ax_arr2[ii].set_ylim([-0.1, 1.1])

        ax_arr[ii].set_title("Level %d" % (3-ii))
        ax_arr2[ii].set_title("Level %d" % (3-ii))

        ax_arr2[ii].legend()

    ax_arr[2].set_xlabel("column position")
    ax_arr2[2].set_xlabel("Time(s)")

    f.suptitle("Column outputs, A_pos=%0.2f, A_len=%0.2f" % (A_pos, A_len), fontsize=16)
    f.suptitle("Column outputs, A_pos=%0.2f, A_len=%0.2f" % (A_pos, A_len), fontsize=16)

    # Plot scaled output of each input column (L4 outputs) ------------------------------
    plt.figure()
    plt.title("Scaled output (L4 outputs ) of each input column")
    ax2 = []
    ax3 = []
    for i in np.arange(7):
        ax2.append(plt.subplot2grid((7, 2), (i, 1)))

    for i in np.arange(3):
        ax3.append(plt.subplot2grid((7, 2), (2*i, 0), rowspan=2))

    for c_idx, column in enumerate(level2_columns):
        ax2[c_idx].set_title("L2 column at %0.2f" % column.i)

        for l4_idx, l4 in enumerate(column.l4_arr):
            ax2[c_idx].plot(t, sim.data[l4.soma_p],
                            label='L1 column at %0.2f'
                                  % column.prev_c_positions[l4_idx])
        ax2[c_idx].legend()
        ax2[c_idx].set_ylim([-0.1, 1.1])

    ax2[c_idx].set_xlabel("Time(s)")

    for c_idx, column in enumerate(level3_columns):
        ax3[c_idx].set_title("L3 column at %0.2f" % column.i)

        for l4_idx, l4 in enumerate(column.l4_arr):
            ax3[c_idx].plot(t, sim.data[l4.soma_p],
                            label='L2 column at %0.2f'
                                  % column.prev_c_positions[l4_idx])
        ax3[c_idx].legend()
        ax3[c_idx].set_ylim([-0.1, 1.1])

    ax3[c_idx].set_xlabel("Time(s)")

    # # Plot mu_i for each column (L4 outputs) ------------------------------
    # plt.figure()
    # plt.title(r"$\mu_{i} \ focus\ for\ column$")
    # ax2 = []
    # ax3 = []
    # for i in np.arange(7):
    #     ax2.append(plt.subplot2grid((7, 2), (i, 1)))
    #
    # for i in np.arange(3):
    #     ax3.append(plt.subplot2grid((7, 2), (2*i, 0), rowspan=2))
    #
    # for c_idx, column in enumerate(level2_columns):
    #
    #     ax2[c_idx].set_title("mu_i Level 2 Column at %0.2f" % column.i)
    #
    #     for l4_idx, l4 in enumerate(column.l4_arr):
    #         ax2[c_idx].plot(t, sim.data[l4.dend_rout_p][:, 0], label="Dendrites at %0.2f" % l4.j)
    #
    #     ax2[c_idx].legend()
    #
    # for c_idx, column in enumerate(level3_columns):
    #
    #     ax3[c_idx].set_title("mu_i Level 3 Column at %0.2f" % column.i)
    #
    #     for l4_idx, l4 in enumerate(column.l4_arr):
    #         ax3[c_idx].plot(t, sim.data[l4.dend_rout_p][:, 0], label="Dendrite at  %0.2f" % l4.j)
    #
    #     ax3[c_idx].legend()
