import numpy as np
import matplotlib.pyplot as plt
import nengo


def routing_function(dend_out):
    mu_i, sigma_att = dend_out

    scale_factor = 0

    if sigma_att != 0:
        scale_factor = np.exp(-(mu_i - prev_level_column_pos) ** 2 / (2 * abs(sigma_att) ** 2))

    return scale_factor


def main(i, rout_pop_size):

    c_spacing = 0.25

    # Common Settings
    tau_ref = 0.002
    tau_rc = 0.02
    tau_psc = 0.1  # post synaptic time constant

    # Network settings
    valid_sf = [1.0, 4/3.0, 2.0, 3.0]
    m = 0.25

    max_mu = max(valid_sf)* i + m
    max_sigma = max(valid_sf) * c_spacing / 2.35

    model = nengo.Network(label='1D Routing Function Evaluation')

    #randomly choose a shift and sampling factor
    s = np.random.choice([-m, 0, m], size=1)
    sf = np.random.choice(valid_sf, size=1)

    with model:
        stim_sf = nengo.Node(sf)
        stim_s = nengo.Node(s)

        # Dendrites Routing
        # d[0] = mu_i = sf * i + s,
        # d[1]= sigma_att =  sf * column_spacing / 2.35
        dend_rout = nengo.Ensemble(
            rout_pop_size,  # population size
            2,    # dimensionality
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=nengo.LIF(tau_ref=tau_ref, tau_rc=tau_rc),
            # neuron_type=nengo.Direct(),
            label='L4 dend rout',
            radius=np.sqrt(max_mu**2 + max_sigma**2)
        )

        # Dendrites
        # d[0] = scaling factor (internal),
        # d[1]= input from previous level connected column
        dend = nengo.Ensemble(
            300,
            1,
            # neuron_type=nengo.Sigmoid(),  # Dendrites do not spike
            neuron_type=nengo.Direct(),
            label='L4 dendrites'
        )

        # Connections
        nengo.Connection(stim_sf, dend_rout[0], transform=i, synapse=tau_psc)
        nengo.Connection(stim_s, dend_rout[0], synapse=tau_psc)

        nengo.Connection(stim_sf, dend_rout[1], transform=c_spacing/2.35, synapse=tau_psc)

        pnts_d0 = np.random.choice(valid_sf, size=1000)  # Valid subsampling values
        pnts_d0 = pnts_d0 * i
        pnts_d0 = pnts_d0 + np.random.choice([-m, 0, m], size=1000)  # Valid shifts
        pnts_d0 = pnts_d0 + np.random.normal(loc=0, scale=0.1, size=1000)

        pnts_d1 = np.random.choice(valid_sf, size=1000) * c_spacing / 2.35 + \
            np.random.normal(loc=0, scale=0.05, size=1000)

        eval_points = np.vstack([pnts_d0, pnts_d1]).T

        nengo.Connection(dend_rout, dend, eval_points=eval_points,
                         synapse=tau_psc, function=routing_function)

        # Probes
        dend_p = nengo.Probe(dend, synapse=0.1)


    sim = nengo.Simulator(model)
    sim.run(5)

    x = routing_function((sf*i + s, sf*c_spacing/2.35))
    rmse = np.sqrt(np.mean((sim.data[dend_p] - x)**2))

    return rmse

if __name__ == '__main__':
    plt.ion()

    rout_pop_size_arr = np.array([100, 500, 1000, 1500, 2000, 3000, 4000])
    avg_error_arr = np.zeros(shape=rout_pop_size_arr.shape[0])

    column_pos = 0
    prev_level_column_pos = 0

    n_runs = 100

    for p_idx, pop_size in enumerate(rout_pop_size_arr):
        rmse_err_arr = np.zeros(shape=n_runs)

        for r_idx in np.arange(n_runs):
            rmse_err_arr[r_idx] = main(column_pos, pop_size)

        avg_error_arr[p_idx] = np.mean(rmse_err_arr)

    plt.figure()
    plt.plot(rout_pop_size_arr, avg_error_arr)



