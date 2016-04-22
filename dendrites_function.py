# -*- coding: utf-8 -*-
"""
The same as dendrites_model.py but without using dendrites.
Models Laminar layer 4 of a cortical column
"""

import numpy as np
import matplotlib.pyplot as plt
import nengo


class Layer4:

    def __init__(self, t_rc, t_ref, t_psc, c_spacing, j, i, m,
                 x_j_hat_node, sf_node, s_node,
                 valid_sf=None, n_dendrites=50):
        """

        :param t_rc:
        :param t_ref:
        :param t_psc: post synaptic time constant
        :param c_spacing: between column spacing
        :param j: position of connected column in previous layer
        :param i: parent column position in c
        :param m: max shift allowed in the column

        :param x_j_hat_node: Nengo node/ensemble of connected column in previous layer
        :param sf_node: Nengo node/ensemble of current layer subsampling factor
        :param s_node: Nengo node/ensemble of relative shift at current level

        Optimization parameters
        :param valid_sf: list of valid subsampling factors. Default = [1, 1.5, 2]
        :param n_dendrites: Number of dendrites for each neuron in L4 soma. Default=50

        :return:
        """

        self.j = j
        self.n_dendrites = n_dendrites

        if valid_sf is None:
            valid_sf = [1, 1.5, 2]

        # Populations --------------------------------------------------------------

        # Dendrites Routing
        # d[0] = mu_i = sf * i + s,
        # d[1]= sigma_att =  sf * column_spacing / 2.35
        max_sf = max(valid_sf)
        max_mu = max_sf * i + m
        max_sigma = max_sf * c_spacing / 2.35

        self.dend_rout = nengo.Ensemble(
            2000,  # population size
            2,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            # neuron_type=nengo.Direct(),
            label='L4 dend rout',
            radius=np.sqrt(max_mu**2 + max_sigma**2)
        )

        # Dendrites
        # d[0] = scaling factor
        # d[1]= input from previous level connected column
        self.dend = nengo.Ensemble(
            300,
            2,
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.Sigmoid(),  # Dendrites do not spike
            radius=np.sqrt(2),
            label='L4 dendrites'
        )

        # Soma
        self.soma = nengo.Ensemble(
            self.n_dendrites,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            label='L4 soma',
        )

        # Output of the routing function can take on a specifiable set of values.
        # When solving for decoders, evaluate the function at points surrounding these values
        # to increase representation accuracy around these points
        pnts_d0 = np.random.choice(valid_sf, size=1000)  # Valid subsampling values
        pnts_d0 = pnts_d0 * i
        pnts_d0 = pnts_d0 + np.random.choice([-m, 0, m], size=1000)  # Valid shifts
        pnts_d0 = pnts_d0 + np.random.normal(loc=0, scale=0.1, size=1000)

        pnts_d1 = np.random.choice(valid_sf, size=1000) * c_spacing / 2.35 + \
            np.random.normal(loc=0, scale=0.1, size=1000)

        eval_points = np.vstack([pnts_d0, pnts_d1]).T

        # Scale factors are also between 0, 1 so use another set of eval points to improve soma
        # representation
        # TODO: Figure out why these eval points doesnt improve representation
        # pnts_d0 = np.random.uniform(0, 1, size=1000)
        # pnts_d1 = np.random.uniform(-1, 1, size=1000)
        # eval_points2 = np.vstack([pnts_d0, pnts_d1]).T

        # Connections ------------------------------------------------
        nengo.Connection(sf_node, self.dend_rout[0], transform=i, synapse=t_psc)
        nengo.Connection(s_node, self.dend_rout[0], synapse=t_psc)
        nengo.Connection(sf_node, self.dend_rout[1], transform=(c_spacing / 2.35), synapse=t_psc)

        nengo.Connection(self.dend_rout, self.dend[0], eval_points=eval_points,
                         function=self.routing_function, synapse=t_psc)
        nengo.Connection(x_j_hat_node, self.dend[1], synapse=t_psc)

        nengo.Connection(self.dend, self.soma, synapse=t_psc,  # eval_points=eval_points2,
                         function=self.scaled_dend_out)

        # Probes --------------------------------------------------------------
        self.dend_rout_p = nengo.Probe(self.dend_rout, synapse=0.1)
        self.dend_p = nengo.Probe(self.dend, synapse=0.1)
        self.soma_p = nengo.Probe(self.soma, synapse=0.1)

    def routing_function(self, dend_out):
        mu_i, sigma_att = dend_out

        scale_factor = 0

        if sigma_att > 0:
            scale_factor = np.exp(-(mu_i - self.j)**2 / (2 * sigma_att**2))

        return scale_factor

    @staticmethod
    def scaled_dend_out(dend_out):
        scale_factor, x_j_hat = dend_out
        return scale_factor * x_j_hat


class TwoDimLayer4:

    def __init__(self, t_rc, t_ref, t_psc, c_spacing, j, i, m,
                 x_j_hat_node, sf_node, s_node,
                 valid_sf=None, n_dendrites=50):
        """

        :param t_rc:
        :param t_ref:
        :param t_psc: post synaptic time constant
        :param c_spacing: between column spacing
        :param j: position of connected column in previous layer
        :param i: parent column position in c
        :param m: max shift allowed in the column

        :param x_j_hat_node: Nengo node/ensemble of connected column in previous layer
        :param sf_node: Nengo node/ensemble of current layer subsampling factor
        :param s_node: Nengo node/ensemble of relative shift at current level

        Optimization parameters
        :param valid_sf: list of valid subsampling factors. Default = [1, 1.5, 2]
        :param n_dendrites: Number of dendrites for each neuron in L4 soma. Default=50

        :return:
        """
        self.j = j
        self.n_dendrites = n_dendrites

        if valid_sf is None:
            valid_sf = [1, 1.5, 2]

        # Populations --------------------------------------------------------------

        # Dendrites Routing
        # d[0] = mu_ix = sf * ix + sx,
        # d[1] = mu_iy = sf * iy + sy,
        # d[2]= sigma_att =  sf * column_spacing / 2.35
        max_sf = max(valid_sf)
        max_mu_x = max_sf * i[0] + m
        max_mu_y = max_sf * i[1] + m
        max_sigma = max_sf * c_spacing

        self.dend_rout = nengo.Ensemble(
            3000,  # population size
            3,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            # neuron_type=nengo.Direct(),
            label='L4 dend rout',
            radius=np.sqrt(max_mu_x**2 + max_mu_y**2 + max_sigma**2)
        )

        # Dendrites
        # d[0] = scaling factor
        # d[1]= input from previous level connected column
        self.dend = nengo.Ensemble(
            300,
            2,
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIFRate(),  # Dendrites do not spike
            # neuron_type=nengo.Direct(),
            radius=np.sqrt(2),
            label='L4 dendrites'
        )

        # Soma
        self.soma = nengo.Ensemble(
            self.n_dendrites,  # population size
            1,  # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=t_ref, tau_rc=t_rc),
            label='L4 soma',
        )

        # Connections ---------------------------------------------------------------------
        nengo.Connection(sf_node, self.dend_rout[0], transform=i[0], synapse=t_psc)
        nengo.Connection(sf_node, self.dend_rout[1], transform=i[1], synapse=t_psc)
        nengo.Connection(sf_node, self.dend_rout[2], transform=(c_spacing / 2.35), synapse=t_psc)

        nengo.Connection(s_node[0], self.dend_rout[0], synapse=t_psc)
        nengo.Connection(s_node[1], self.dend_rout[1], synapse=t_psc)

        # Output of the routing function can take on a specifiable set of values.
        # When solving for decoders, evaluate the function at points surrounding these values
        # to increase representation accuracy around these points
        sample_sf = np.random.choice(valid_sf, size=2000)
        sample_m = np.random.choice([-m, 0, m], size=2000)

        pnts_d0 = sample_sf[0: 1000] * i[0] + sample_m[0: 1000] + \
            np.random.normal(loc=0, scale=0.1, size=1000)
        pnts_d1 = sample_sf[0: 1000] * i[1] + sample_m[0: 1000] + \
            np.random.normal(loc=0, scale=0.1, size=1000)
        pnts_d2 = np.random.choice(valid_sf, size=1000) * c_spacing / 2.35 + \
            np.random.normal(loc=0, scale=0.1, size=1000)

        eval_points = np.vstack([pnts_d0, pnts_d1, pnts_d2]).T

        nengo.Connection(self.dend_rout, self.dend[0], eval_points=eval_points,
                         function=self.routing_function, synapse=t_psc)

        nengo.Connection(x_j_hat_node, self.dend[1], synapse=t_psc)

        nengo.Connection(self.dend, self.soma, synapse=t_psc,
                         function=self.scaled_dend_out)

        # Probes --------------------------------------------------------------
        self.dend_rout_p = nengo.Probe(self.dend_rout, synapse=0.1)
        self.dend_p = nengo.Probe(self.dend, synapse=0.1)
        self.soma_p = nengo.Probe(self.soma, synapse=0.1)

    def routing_function(self, dend_out):
        mu_ix, mu_iy, sigma_att = dend_out

        scale_factor = 0

        if sigma_att > 0:
            scale_factor = np.exp(-((mu_ix - self.j[0])**2 + (mu_iy - self.j[1])**2) /
                                  (2 * sigma_att**2))
        return scale_factor

    @staticmethod
    def scaled_dend_out(dend_out):
        scale_factor, x_j_hat = dend_out
        return scale_factor * x_j_hat


if __name__ == '__main__':
    plt.ion()

    tau_rc = 0.02
    tau_ref = 0.002
    tau_psc = 0.1

    between_column_dist = 0.25
    column_m = 1 * between_column_dist
    valid_sf_values_for_level = [1, 1.5, 2]

    # # -----------------------------------------------------------------------------
    # # Single Layer 4 model
    # # -----------------------------------------------------------------------------
    # # Build the Network
    # model = nengo.Network(label='Single Layer 4')
    #
    # column_pos = 0
    # prev_level_column_pos = 0
    #
    # # Inputs to dendrites
    # sf = 1
    # rel_shift = 0
    # prev_level_column_out = 1
    #
    # with model:
    #     stim_sf = nengo.Node(sf)
    #     stim_s = nengo.Node(rel_shift)
    #     stim_x_j_hat = nengo.Node(prev_level_column_out)
    #
    #     l4 = Layer4(
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
    # f1.suptitle("Column pos %0.2f, previous layer connected pos %0.2f"
    #             % (column_pos, prev_level_column_pos))
    #
    # ax_arr[0].set_title("Dendrites Routing")
    # ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 0],
    #                label='L4 Dend D0 (Expected %0.2f)' % (sf * column_pos + rel_shift))
    # ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 1],
    #                label='L4 Dend D1 (Expected %0.2f)' % (sf * between_column_dist / 2.35))
    # ax_arr[0].legend()
    #
    # ax_arr[1].set_title("Dendrites out")
    # ax_arr[1].plot(
    #     t, sim.data[l4.dend_p][:, 0],
    #     label='Scale factor(Expected=%0.2f)'
    #           % (l4.routing_function((sf*column_pos + rel_shift, sf*between_column_dist/2.35))))
    # ax_arr[1].plot(t, sim.data[l4.dend_p][:, 1], label='Input from previous column')
    # ax_arr[1].plot(t, sim.data[l4.soma_p], label='L4 soma output')
    # ax_arr[1].legend()

    # -----------------------------------------------------------------------------
    # Multiple Layer 4s example
    # -----------------------------------------------------------------------------
    # Each layer 4 model is connected to a single previous level cortical column,
    # Each Cortical column is connected to multiple previous level columns and so will
    #  have multiple layer 4 populations.

    # Build the Network
    model = nengo.Network(label='Multiple Layer 4')

    column_pos = 0.25

    sf = 1
    rel_shift = -0.25

    prev_level_column_pos = np.array([0, 0.25, 0.5])
    prev_level_column_out = np.array([1, 1, 1])

    with model:
        stim_sf = nengo.Node(sf)
        stim_s = nengo.Node(rel_shift)
        stim_x_j_hat = nengo.Node(prev_level_column_out)

        l4_arr = []
        for p_idx, pos in enumerate(prev_level_column_pos):
            l4 = Layer4(
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
        f1.suptitle("Column pos %0.2f, previous layer connected pos %0.2f [sf=%0.2f, s=%0.2f]"
                    % (column_pos, prev_level_column_pos[l4_idx], sf, rel_shift))

        ax_arr[0].set_title("Dendrites Routing")
        ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 0],
                       label='L4 Dend D0 (mu_i = %0.2f)' % (sf * column_pos + rel_shift))
        ax_arr[0].plot(t, sim.data[l4.dend_rout_p][:, 1],
                       label='L4 Dend D1 (sigma_att = %0.2f)' % (sf * between_column_dist / 2.35))
        ax_arr[0].legend()

        ax_arr[1].set_title("Dendrites 0ut")
        ax_arr[1].plot(
            t, sim.data[l4.dend_p][:, 0],
            label='Scale factor(Expected=%0.2f)' % (l4.routing_function(
                (sf*column_pos + rel_shift, sf*between_column_dist/2.35))))
        ax_arr[1].plot(t, sim.data[l4.dend_p][:, 1], label='Input from previous column')
        ax_arr[1].plot(t, sim.data[l4.soma_p], label='L4 soma output')

        ax_arr[1].legend()

    f, ax_arr = plt.subplots()
    ax_arr.set_title("Column output")
    ax_arr.plot(t, sim.data[l2_3_output_p], label='l2/3 output')
    ax_arr.legend()
