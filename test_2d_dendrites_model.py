# -*- coding: utf-8 -*-
"""
This Files tests the 2 input dendrite function
"""

import numpy as np
import matplotlib.pyplot as plt
import nengo

import dendrites_function as dend_func

reload(dend_func)

if __name__ == '__main__':
    plt.ion()

    tau_rc = 0.02
    tau_ref = 0.002
    tau_psc = 0.1

    between_column_dist = 0.25
    column_m = 1 * between_column_dist
    valid_sf_values_for_level = [1, 1.5, 2]

    # # -----------------------------------------------------------------------------
    # # Single 2D Layer 4 model
    # # -----------------------------------------------------------------------------
    # # Build the Network
    # model = nengo.Network(label='Single Layer 4 (2D)')
    #
    # column_pos = np.array([0, 0])
    # prev_level_column_pos = ([0, 0])
    #
    # # Inputs to dendrites
    # sf = 1
    # rel_shift = np.array([-0.25, -0.25])
    # prev_level_column_out = 1
    #
    # with model:
    #     stim_sf = nengo.Node(sf)
    #     stim_s = nengo.Node(rel_shift)
    #     stim_x_j_hat = nengo.Node(prev_level_column_out)
    #
    #     l4 = dend_func.TwoDimLayer4(
    #         t_rc=tau_rc,
    #         t_ref=tau_ref,
    #         t_psc=tau_psc,
    #         c_spacing=between_column_dist,
    #         i=column_pos,
    #         j=prev_level_column_pos,
    #         x_j_hat_node=stim_x_j_hat,
    #         sf_node=stim_sf,
    #         s_node=stim_s,
    #         m=column_m,
    #         valid_sf=valid_sf_values_for_level
    #     )
    #
    # # Run the model and check output
    # sim = nengo.Simulator(model)
    # sim.run(5)
    # t = sim.trange()
    #
    # f1, ax_arr = plt.subplots(2, 1)
    # f1.suptitle("Column pos %s, previous layer connected pos %s"
    #             % (column_pos, prev_level_column_pos))
    #
    # ax_arr[0].set_title("Dendrites Routing")
    # ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 0],
    #                label='L4 Dend D0 (mu_ix %0.2f)' % (sf * column_pos[0] + rel_shift[0]))
    # ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 1],
    #                label='L4 Dend D0 (mu_iy %0.2f)' % (sf * column_pos[0] + rel_shift[1]))
    # ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 2],
    #                label='L4 Dend D1 (sigma_att %0.2f)' % (sf * between_column_dist / 2.35))
    #
    # ax_arr[0].legend()
    #
    # ax_arr[1].set_title("Dendrites out")
    # ax_arr[1].plot(
    #     t, sim.data[l4.dend_p][:, 0],
    #     label='Scale factor(Expected=%0.2f)'
    #           % (l4.routing_function((sf*column_pos[0] + rel_shift[0],
    #                                   sf*column_pos[1] + rel_shift[1],
    #                                   sf*between_column_dist/2.35))))
    # ax_arr[1].plot(t, sim.data[l4.dend_p][:, 1], label='Input from previous column')
    # ax_arr[1].plot(t, sim.data[l4.soma_p], label='L4 soma output')
    # ax_arr[1].legend()

    # -----------------------------------------------------------------------------
    # Multiple 2D Layer 4s example
    # -----------------------------------------------------------------------------
    # Each layer 4 model is connected to a single previous level cortical column,
    # Each Cortical column is connected to multiple previous level columns and so will
    #  have multiple layer 4 populations.

    # Build the Network
    model = nengo.Network(label='Multiple 2D Layer 4')
    a = 3  # This is a radius, in total there are a^2 number of connections

    column_pos = np.array([0, 0])

    # To make life easier just assume a fixed pattern of mapping between 1D handling and actual
    # 2D representation. indexes starts from bottom right
    offsets = (np.arange(a) - int(a)/2) * between_column_dist
    prev_level_column_pos = [(column_pos[0] + x, column_pos[1] + y)
                             for x in offsets for y in offsets]

    # mapping should follow the same indexing as the prev_level_column_pos
    prev_level_column_out = np.ones(shape=a**2)

    sf = 1
    rel_shift = np.array([0.25, 0.25])

    with model:
        stim_sf = nengo.Node(sf)
        stim_s = nengo.Node(rel_shift)
        stim_x_j_hat = nengo.Node(prev_level_column_out)

        l4_arr = []
        for p_idx, pos in enumerate(prev_level_column_pos):
            l4 = dend_func.TwoDimLayer4(
                t_rc=tau_rc,
                t_ref=tau_ref,
                t_psc=tau_psc,
                c_spacing=between_column_dist,
                i=column_pos,
                j=pos,
                x_j_hat_node=stim_x_j_hat[p_idx],
                sf_node=stim_sf,
                s_node=stim_s,
                m=column_m,
                valid_sf=valid_sf_values_for_level,
            )

            l4_arr.append(l4)

        # Simulate Layer 2/3 of the cortical column
        l2_3_output = nengo.Ensemble(
            100,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='L2/3  out',
        )

        for l4_idx, l4 in enumerate(l4_arr):
            nengo.Connection(l4.soma, l2_3_output, synapse=tau_psc)

        l2_3_output_p = nengo.Probe(l2_3_output, synapse=0.1)

    # Run the model and check output
    sim = nengo.Simulator(model)
    sim.run(5)
    t = sim.trange()

    for l4_idx, l4 in enumerate(l4_arr):
        f1, ax_arr = plt.subplots(2, 1)
        f1.suptitle("Column pos %s, previous layer connected pos %s [sf=%0.2f, s=%s]"
                    % (column_pos, prev_level_column_pos[l4_idx], sf, rel_shift))

        ax_arr[0].set_title("Dendrites Routing")
        ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 0],
                       label='L4 Dend D0 (mu_ix = %0.2f)' % (sf * column_pos[0] + rel_shift[0]))
        ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 1],
                       label='L4 Dend D0 (mu_iy = %0.2f)' % (sf * column_pos[1] + rel_shift[1]))
        ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 2],
                       label='L4 Dend D1 (sigma_att = %0.2f)' % (sf * between_column_dist / 2.35))
        ax_arr[0].legend()

        ax_arr[1].set_title("Dendrites 0ut")
        ax_arr[1].plot(
            t, sim.data[l4.dend_p][:, 0],
            label='Scale factor(Expected=%0.2f)' % (l4.routing_function(
                (sf*column_pos[0] + rel_shift[0],
                 sf*column_pos[1] + rel_shift[1],
                 sf*between_column_dist/2.35))))
        ax_arr[1].plot(t, sim.data[l4.dend_p][:, 1], label='Input from previous column')
        ax_arr[1].plot(t, sim.data[l4.soma_p], label='L4 soma output')

        ax_arr[1].legend()
