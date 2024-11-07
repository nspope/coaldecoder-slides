import tskit
import os
import pickle
import numpy as np
import msprime

force_overwrite = True


def get_model():
    time_step = np.linspace(0, 50000, 1000)
    mean = 1e4
    ampl = 5e3
    phase = 0
    freq = 1e4
    ne_traj = mean + ampl * np.cos(2*np.pi*(time_step - phase) / freq)
    demog = msprime.Demography.isolated_model([ne_traj[0]])
    for t, ne in zip(time_step, ne_traj):
        demog.add_population_parameters_change(time=t, initial_size=ne, population=0)
    return demog


def get_pop_size_and_coal_rate():
    if os.path.exists("tmp.rates.p") and not force_overwrite:
        return pickle.load(open("tmp.rates.p", "rb"))
    else:
        demog = get_model()
        time_step = np.linspace(0, 50000, 1000)
        debug = demog.debug()
        coal_rate, coal_cdf = debug.coalescence_rate_trajectory(lineages={"pop_0" : 2}, steps=time_step)
        coal_cdf = 1 - coal_cdf
        pop_size = debug.population_size_trajectory(steps=time_step)
        out = (time_step, coal_rate, coal_cdf, pop_size)
        pickle.dump(out, open("tmp.rates.p", "wb"))
        return out


def get_ts() -> tskit.TreeSequence:
    if os.path.exists("tmp.trees") and not force_overwrite:
        return tskit.load("tmp.trees")
    else:
        model = get_model()
        ts = msprime.sim_ancestry(samples=50, demography=model, recombination_rate=1e-8, sequence_length=2e6, random_seed=1)
        ts.dump("tmp.trees")
        return ts


def average_ecdf(ts, cumulative=True):
    node_weights = ts.pair_coalescence_counts(pair_normalise=True)
    node_times = np.unique(ts.nodes_time)
    node_index = np.digitize(ts.nodes_time, node_times) - 1
    assert np.all(np.logical_and(node_index >= 0, node_index < node_times.size))
    node_weights = np.bincount(node_index, weights=node_weights, minlength=node_times.size)
    if cumulative:
        node_weights = np.cumsum(node_weights)
        assert np.isclose(node_weights[-1], 1.0)
    return node_times, node_weights 


def get_ne_est(ts, grid):
    times, weights = average_ecdf(ts, cumulative=False)
    bins = np.digitize(times, grid) - 1
    assert np.all(bins > -1)
    ecdf = np.bincount(bins, weights=weights)
    ecdf = np.append(0, np.cumsum(ecdf))
    assert np.isclose(ecdf[-1], 1.0)
    ecdf = ecdf[:-1]
    assert ecdf.size == grid.size
    phi = (ecdf[1:] - ecdf[:-1]) / (1 - ecdf[:-1])
    return -np.log(1 - phi) / np.diff(grid)


