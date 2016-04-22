# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import nengo

import dendrites_model as dend
import dendrites_function as dend_func

reload(dend)
reload(dend_func)


class CorticalColumn:

    def __init__(self, t_rc, t_ref, t_psc, c_spacing, m, m_prev, i, prev_c_positions,
                 size_max, size_curr_l, size_prev_l,
                 prev_c_out_nodes, l5_pos_node, l5_size_node,
                 valid_sf=None):

        self.m = m
        self.m_prev = m_prev
        self.prev_c_positions = prev_c_positions

        self.size_max = size_max
        self.size_curr_l = size_curr_l
        self.size_prev_l = size_prev_l
        self.i = i

        if valid_sf is None:
            valid_sf = [1, 1.5, 2]

        # Populations ---------------------------------------------------------------

        # Layer 6 relative shift from previous to current level
        self.l6_shift = nengo.Ensemble(
            300,  # population size
            1,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            # neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            neuron_type=nengo.LIF(),
            label='Layer 6 relative shift',
        )

        # Layer 6 subsampling factor
        self.l6_sf = nengo.Ensemble(
            300,  # population size
            1,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            # neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            neuron_type=nengo.LIF(),
            label='Layer 6 sampling factor',
            radius=max(valid_sf)
        )

        # Layer 5 local representation of global signals are common for all columns in a level
        # and are assumed to be represented outside the column class.

        # Layer 4 populations (one for each connected column)
        self.l4_arr = []
        for p_idx, pos in enumerate(prev_c_positions):
            l4 = dend_func.Layer4(
                t_rc=t_rc,
                t_ref=t_ref,
                t_psc=t_psc,
                c_spacing=c_spacing,
                i=self.i,
                j=pos,
                x_j_hat_node=prev_c_out_nodes[p_idx],
                sf_node=self.l6_sf,
                s_node=self.l6_shift,
                m=self.m,
                valid_sf=valid_sf,
            )

            self.l4_arr.append(l4)

        # Layer 2/3 aggregate scaled feed forward inputs from all Layer 4 populations
        self.l2_3_output = nengo.Ensemble(
            300,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            label='L2/3 out',
        )

        # Connections ----------------------------------------------------------
        nengo.Connection(l5_pos_node, self.l6_shift,
                         function=self.l6_relative_shift, synapse=t_psc)

        # Size cannot be negative optimize the representation accuracy over non negative values
        eval_points = np.random.uniform(0, 1, size=(1000, 1)) + \
            np.random.normal(loc=0, scale=0.1)  # size cannot be negative
        nengo.Connection(l5_size_node, self.l6_sf, eval_points=eval_points,
                         function=self.l6_sampling_factor, synapse=t_psc)

        for l4_idx, l4 in enumerate(self.l4_arr):
            nengo.Connection(l4.soma, self.l2_3_output, synapse=t_psc)

        # Probes
        self.l6_shift_p = nengo.Probe(self.l6_shift, synapse=0.1)
        self.l6_sf_p = nengo.Probe(self.l6_sf, synapse=0.1)
        self.l2_3_output_p = nengo.Probe(self.l2_3_output, synapse=0.1)

    def l6_relative_shift(self, l5_pos_out):
        tgt_pos, theta_c = l5_pos_out

        if abs(tgt_pos) <= self.m_prev:
            theta_prev = 0
        else:
            theta_prev = tgt_pos - np.sign(tgt_pos) * self.m_prev

        return theta_prev - theta_c

    def l6_sampling_factor(self, tgt_size):
        return (min(max(tgt_size*self.size_max, self.size_curr_l), self.size_prev_l) - 1) / \
             (self.size_curr_l - 1)


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
    size_curr_level = 3.
    size_prev_level = 5.
    max_size = 5.0

    # External inputs
    A_len = 3.0 / 5.0
    A_pos = -0.25
    theta_curr = 0
    prev_level_column_out = np.array([1, 1, 1])

    # Constants for the column
    column_pos = 0.25
    prev_level_column_pos = np.array([0, 0.25, 0.5])

    # Build the Network
    model = nengo.Network(label='Cortical Column')

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
            300,  # population size
            2,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            label='Layer 5 position, theta_curr',
            radius=np.sqrt(2)
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
        nengo.Connection(stim_g_pos, l5_pos[0], synapse=tau_psc)
        nengo.Connection(stim_theta_curr, l5_pos[1], synapse=tau_psc)
        nengo.Connection(stim_g_len, l5_size, synapse=tau_psc)

        # Probes
        l5_pos_p = nengo.Probe(l5_pos, synapse=0.1)
        l5_size_p = nengo.Probe(l5_size, synapse=0.1)

        column1 = CorticalColumn(
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
    ax_arr[0].plot(t, sim.data[l5_pos_p][:, 0], label='Apos=%0.2f' % A_pos)
    ax_arr[0].plot(t, sim.data[l5_pos_p][:, 1], label=r'$\theta_{curr}=%0.2f$' % theta_curr)
    ax_arr[0].plot(t, sim.data[l5_size_p], label='Asize=%0.2f' % A_len)
    ax_arr[0].set_title("Feedback Inputs")
    ax_arr[0].legend()

    # Cortical Column Details
    ax_arr[1].plot(t, sim.data[column1.l6_shift_p],
                   label='s=%0.2f' % column1.l6_relative_shift((A_pos, theta_curr)))
    ax_arr[1].plot(t, sim.data[column1.l6_sf_p],
                   label='sf=%0.2f' % column1.l6_sampling_factor(A_len))
    ax_arr[1].plot(t, sim.data[column1.l2_3_output_p], label='Column Output')
    ax_arr[1].set_title("Column Values")
    ax_arr[1].legend()

    # Dendrites output
    for idx, layer4 in enumerate(column1.l4_arr):
        ax_arr[2].plot(t, sim.data[layer4.soma_p],
                       label="l4 pos %0.2f output" % (prev_level_column_pos[idx]))

    ax_arr[2].set_title("Dendrites")
    ax_arr[2].legend()

    sf = column1.l6_sampling_factor(A_len)
    rel_shift = column1.l6_relative_shift((A_pos, theta_curr))
    f.suptitle("sf=%0.2f, s=%0.2f, column position=%0.2f" % (sf, rel_shift, column_pos))

    # ------------------------------------------------------------------------------------
    sf = column1.l6_sampling_factor(A_len)
    rel_shift = column1.l6_relative_shift((A_pos, theta_curr))
    print("Relative shift%0.2f, column_pos %0.2f" % (rel_shift, column_pos))

    for idx, layer4 in enumerate(column1.l4_arr):
        f1, ax_arr = plt.subplots(2, 1)
        f1.suptitle("Column pos %0.2f, previous layer connected pos %0.2f"
                    % (column_pos, prev_level_column_pos[idx]))

        ax_arr[0].set_title("Dendrites Routing")
        ax_arr[0].plot(t, sim.data[layer4.dend_rout_p][:, 0],
                       label='L4 Dend D0 (Expected %0.2f)' % (sf * column_pos + rel_shift))
        ax_arr[0].plot(t, sim.data[layer4.dend_rout_p][:, 1],
                       label='L4 Dend D1 (Expected %0.2f)' % (sf * between_column_dist / 2.35))
        ax_arr[0].legend()

        ax_arr[1].set_title("Dendrites out")
        ax_arr[1].plot(
            t, sim.data[layer4.dend_p][:, 0],
            label='Scale factor(Expected=%0.2f)' % (layer4.routing_function(
                (sf*column_pos + rel_shift, sf*between_column_dist/2.35))))
        ax_arr[1].plot(t, sim.data[layer4.dend_p][:, 1], label='Input from previous column')
        ax_arr[1].plot(t, sim.data[layer4.soma_p], label='L4 soma output')

        ax_arr[1].legend()
