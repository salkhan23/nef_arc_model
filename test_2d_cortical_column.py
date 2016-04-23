# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import nengo

import cortical_column as cc

reload(cc)

if __name__ == '__main__':
    plt.ion()

    # Common Settings
    tau_ref = 0.002
    tau_rc = 0.02
    tau_psc = 0.1  # post synaptic time constant

    valid_sampling_factors = [1, 1.5, 2]
    between_column_dist = 0.25
    m_curr_level = 1 * between_column_dist  # maximum shift of 1 column for this level
    m_prev_level = 0

    a = 3  # This is a radius, in total there are a^2 number of connections

    size_curr_level = np.array([3, 3])
    size_prev_level = np.array([5, 5])

    max_size = 5.0

    # External inputs
    A_len = 3.0 / 5.0
    A_pos = np.array([0.25, 0.25])
    theta_curr = np.array([0, 0])

    # Constants for the column
    column_pos = ([0.25, 0.25])

    # To make life easier just assume a fixed pattern of mapping between 1D handling and actual
    # 2D representation. Indexes starts from bottom right
    offsets = (np.arange(a) - int(a)/2) * between_column_dist
    prev_level_column_pos = [(column_pos[0] + x, column_pos[1] + y)
                             for x in offsets for y in offsets]

    # mapping should follow the same indexing as the prev_level_column_pos
    prev_level_column_out = np.ones(shape=a**2)

    # Build the Network
    model = nengo.Network(label='2D Cortical Column')

    with model:

        # ------------------------------------------------------------------------
        # Current Level Constants
        # -----------------------------------------------------------------------
        # These are external to the column and are constant for the current level
        stim_g_pos = nengo.Node(A_pos)
        stim_g_len = nengo.Node(A_len)
        stim_theta_curr = nengo.Node(theta_curr)

        # Feed forward inputs from previous level columns
        stim_x_j_hat = nengo.Node(prev_level_column_out)

        # Layer 5 position - locally represents global position and theta at current level
        l5_pos = nengo.Ensemble(
            900,  # population size
            4,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 position, theta_curr',
            radius=np.sqrt(4)
        )

        # Layer 5 size - locally represent global size
        l5_size = nengo.Ensemble(
            300,  # population size
            1,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 size',
        )

        # Connections
        nengo.Connection(stim_g_pos[0], l5_pos[0], synapse=tau_psc)
        nengo.Connection(stim_g_pos[1], l5_pos[1], synapse=tau_psc)

        nengo.Connection(stim_theta_curr[0], l5_pos[2], synapse=tau_psc)
        nengo.Connection(stim_theta_curr[1], l5_pos[3], synapse=tau_psc)

        nengo.Connection(stim_g_len, l5_size, synapse=tau_psc)

        # Probes
        l5_pos_p = nengo.Probe(l5_pos, synapse=0.1)
        l5_size_p = nengo.Probe(l5_size, synapse=0.1)

        column1 = cc.TwoDimCorticalColumn(
            t_ref=tau_ref, t_rc=tau_rc, t_psc=tau_psc,
            c_spacing=between_column_dist,
            m=m_curr_level,
            m_prev=m_prev_level,
            i=column_pos,
            prev_c_positions=prev_level_column_pos,
            size_max=max_size,
            size_curr_l=size_curr_level,
            size_prev_l=size_prev_level,
            prev_c_out_nodes=stim_x_j_hat,
            l5_pos_node=l5_pos,
            l5_size_node=l5_size,
            valid_sf=valid_sampling_factors
        )

    # -----------------------------------------------------------------------------------
    sim = nengo.Simulator(model)
    sim.run(5)
    t = sim.trange()

    f, ax_arr = plt.subplots(3, 1)

    # Inputs
    ax_arr[0].plot(t, sim.data[l5_pos_p][:, 0], label='Apos_x=%0.2f' % A_pos[0])
    ax_arr[0].plot(t, sim.data[l5_pos_p][:, 1], label='Apos_y=%0.2f' % A_pos[1])

    ax_arr[0].plot(t, sim.data[l5_pos_p][:, 2], label=r'$\theta_{curr_x}=%0.2f$' % theta_curr[0])
    ax_arr[0].plot(t, sim.data[l5_pos_p][:, 3], label=r'$\theta_{curr_y}=%0.2f$' % theta_curr[1])

    ax_arr[0].plot(t, sim.data[l5_size_p], label='Asize=%0.2f' % A_len)
    ax_arr[0].set_title("Feedback Inputs")
    ax_arr[0].legend()

    # Cortical Column Details
    shift_x, shift_y = \
        column1.l6_relative_shift((A_pos[0], A_pos[1], theta_curr[0], theta_curr[1]))
    ax_arr[1].plot(t, sim.data[column1.l6_shift_p][:, 0], label='shift_x=%0.2f' % shift_x)
    ax_arr[1].plot(t, sim.data[column1.l6_shift_p][:, 1], label='shift_y=%0.2f' % shift_y)
    ax_arr[1].plot(t, sim.data[column1.l6_sf_p],
                   label='sf=%0.2f' % column1.l6_sampling_factor(A_len))
    ax_arr[1].plot(t, sim.data[column1.l2_3_output_p], label='Column Output')
    ax_arr[1].set_title("Column Calculations")
    ax_arr[1].legend()

    # Layer 4 output
    for idx, layer4 in enumerate(column1.l4_arr):
        ax_arr[2].plot(t, sim.data[layer4.soma_p],
                       label="l4 pos {0} output".format(prev_level_column_pos[idx]))

    ax_arr[2].set_title("L4 outputs")
    ax_arr[2].legend()

    sf = column1.l6_sampling_factor(A_len)
    f.suptitle("sf=%0.2f, s=[%0.2f, %0.2f], column position=%s"
               % (sf, shift_x, shift_y, column_pos))

    # ------------------------------------------------------------------------------------
    sf = column1.l6_sampling_factor(A_len)
    shift_x, shift_y = \
        column1.l6_relative_shift((A_pos[0], A_pos[1], theta_curr[0], theta_curr[1]))
    print("Relative shift [%0.2f, %0.2f], column_pos %s" % (shift_x, shift_y, column_pos))

    for idx, layer4 in enumerate(column1.l4_arr):
        f1, ax_arr = plt.subplots(2, 1)
        f1.suptitle("Column pos %s, previous layer connected pos %s"
                    % (column_pos, prev_level_column_pos[idx]))

        ax_arr[0].set_title("Dendrites Routing")
        ax_arr[0].plot(t, sim.data[layer4.dend_rout_p][:, 0],
                       label='L4 Dend D0 (Expected %0.2f)' % (sf * column_pos[0] + shift_x))
        ax_arr[0].plot(t, sim.data[layer4.dend_rout_p][:, 1],
                       label='L4 Dend D0 (Expected %0.2f)' % (sf * column_pos[1] + shift_y))
        ax_arr[0].plot(t, sim.data[layer4.dend_rout_p][:, 2],
                       label='L4 Dend D1 (Expected %0.2f)' % (sf * between_column_dist / 2.35))
        ax_arr[0].legend()

        ax_arr[1].set_title("Dendrites out")
        ax_arr[1].plot(
            t, sim.data[layer4.dend_p][:, 0],
            label='Scale factor(Expected=%0.2f)' % (layer4.routing_function(
                (sf*column_pos[0] + shift_x,
                 sf*column_pos[1] + shift_y,
                 sf*between_column_dist/2.35))))

        ax_arr[1].plot(t, sim.data[layer4.dend_p][:, 1], label='Input from previous column')
        ax_arr[1].plot(t, sim.data[layer4.soma_p], label='L4 soma output')

        ax_arr[1].legend()
