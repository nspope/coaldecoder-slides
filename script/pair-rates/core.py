import numpy as np
import tskit
import os
import pickle
import nlopt
from coaldecoder import TrioCoalescenceRateModel, PairCoalescenceRateModel
from coaldecoder import TrioCoalescenceRates, PairCoalescenceRates

import matplotlib


def rates_and_demography(
        time_grid,
        intercept=[1e5, 1e5], 
        phase=[0, 5e3], 
        frequency=[1/1e4, 1/1e4], 
        amplitude=[9e4, 9e4], 
        pulse_mode=[2e4, 4e4], 
        pulse_sd=[1e3, 1e3], 
        #pulse_on=[1e-4, 1e-3], 
        pulse_on=[5e-4, 2e-3], 
        pulse_off=[1e-6, 1.1e-6], 
        pairs_only=True,
    ):
    """
    Pair rates for two oscillating populations with pulse migration
    """

    grid_size = time_grid.size
    epoch_start = time_grid[:-1]
    demographic_parameters = np.zeros((3, 3, grid_size - 1))
    demographic_parameters[0,0] = intercept[0] + amplitude[0] * np.cos(2 * np.pi * (epoch_start + phase[0]) * frequency[0])
    demographic_parameters[0,1] = pulse_off[0] + np.exp(-(epoch_start - pulse_mode[0]) ** 2 / pulse_sd[0] ** 2) * (pulse_on[0] - pulse_off[0])
    demographic_parameters[1,0] = pulse_off[1] + np.exp(-(epoch_start - pulse_mode[1]) ** 2 / pulse_sd[1] ** 2) * (pulse_on[1] - pulse_off[1])
    demographic_parameters[1,1] = intercept[1] + amplitude[1] * np.cos(2 * np.pi * (epoch_start + phase[1]) * frequency[1])
    demographic_parameters[2,2] = np.inf
    admixture_coefficients = np.zeros((3, 3, grid_size - 1))
    for i in range(grid_size - 1): admixture_coefficients[:, :, i] = np.eye(3, 3)

    decoder = TrioCoalescenceRateModel(3)
    expected_rates = decoder.forward(demographic_parameters, admixture_coefficients, np.diff(time_grid))

    state_labels = np.array(decoder.labels(['0','1','2']))
    pair_subset = np.isin(state_labels, ["t1::((0,0),2)", "t1::((0,1),2)", "t1::((1,1),2)"])
    trio_subset = np.isin(state_labels, 
        ["t1::((0,0),0)", "t1::((0,0),1)", "t1::((0,1),0)", "t1::((0,1),1)", "t1::((1,1),0)", "t1::((1,1),1)"] +
        ["t2::((0,0),0)", "t2::((0,0),1)", "t2::((0,1),0)", "t2::((0,1),1)", "t2::((1,1),0)", "t2::((1,1),1)"]
    )
    subset = pair_subset if pairs_only else trio_subset

    return np.diff(time_grid), expected_rates[subset], demographic_parameters[:2, :2]


def plot_ne(ne_ax, duration, params):
    start = np.cumsum(np.append(0, duration))[:-1]
    ne_ax.plot(start / 1e3, params[0,0], label=r"$N_{A}$", color="dodgerblue")
    ne_ax.plot(start / 1e3, params[1,1], label=r"$N_{B}$", color="firebrick")
    ne_ax.set_ylim(8e3, 8e5)
    ne_ax.set_yscale('log')
    ne_ax.set_ylabel("Haploid $N_e$")
    ne_ax.legend(ncol=2, loc='upper left')


def plot_migr(mi_ax, duration, params):
    start = np.cumsum(np.append(0, duration))[:-1]
    mi_ax.plot(start / 1e3, params[0,1], label=r"$M_{A \rightarrow B}$", color="dodgerblue", linestyle='dashed')
    mi_ax.plot(start / 1e3, params[1,0], label=r"$M_{B \rightarrow A}$", color="firebrick", linestyle='dashed')
    mi_ax.set_yscale('log')
    mi_ax.set_ylabel("Migration rate")
    mi_ax.legend(ncol=1, loc='upper left')


def plot_rates(ra_ax, duration, rates, pairs_only=True):
    start = np.cumsum(np.append(0, duration))[:-1]
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    for i, label in enumerate(rate_names):
        ra_ax.plot(start / 1e3, rates[i], label=label)
    ra_ax.set_yscale('log')
    ra_ax.set_ylabel("Pair coalescence rate")
    ra_ax.legend(ncol=1, loc='lower right')


def add_highlight(axs, highlight):
    rect = matplotlib.patches.Rectangle((highlight[0], 1e-30), highlight[1] - highlight[0], 1e30, fc = 'gray', alpha=0.3)
    if highlight is not None:
        axs.add_patch(rect)
    return rect
