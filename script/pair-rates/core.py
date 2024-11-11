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


def plot_ne(ne_ax, duration, params, line_kwargs={}):
    start = np.cumsum(np.append(0, duration))[:-1]
    ne_ax.plot(start / 1e3, params[0,0], label=r"$N_{A}$", color="dodgerblue", **line_kwargs)
    ne_ax.plot(start / 1e3, params[1,1], label=r"$N_{B}$", color="firebrick", **line_kwargs)
    ne_ax.set_yscale('log')
    ne_ax.set_ylabel("Haploid $N_e$")
    ne_ax.legend(ncol=2, loc='upper left')


def plot_migr(mi_ax, duration, params, line_kwargs={}):
    start = np.cumsum(np.append(0, duration))[:-1]
    mi_ax.plot(start / 1e3, params[0,1], label=r"$M_{A \rightarrow B}$", color="dodgerblue", **line_kwargs)
    mi_ax.plot(start / 1e3, params[1,0], label=r"$M_{B \rightarrow A}$", color="firebrick", **line_kwargs)
    mi_ax.set_yscale('log')
    mi_ax.set_ylabel("Migration rate")
    mi_ax.legend(ncol=1, loc='upper left')


def plot_rates(ra_ax, duration, rates, colors=None, pairs_only=True, draw_legend=True, line_kwargs={}):
    start = np.cumsum(np.append(0, duration))[:-1]
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    if colors is None:
        colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [colors[i % len(colors)] for i in range(len(rate_names))]
    for i, label in enumerate(rate_names):
        ra_ax.plot(start / 1e3, rates[i], label=label, color=colors[i], **line_kwargs)
    ra_ax.set_yscale('log')
    ra_ax.set_ylabel("Pair coalescence rate")
    if draw_legend:
        ra_ax.legend(ncol=1, loc='lower right')


def add_highlight(axs, highlight):
    rect = matplotlib.patches.Rectangle((highlight[0], 1e-30), highlight[1] - highlight[0], 1e30, fc = 'gray', alpha=0.3)
    if highlight is not None:
        axs.add_patch(rect)
    return rect


def calculate_gradient(target_rates, demographic_parameters, duration, pairs_only=True):
    P = demographic_parameters.shape[0]
    weights = np.ones_like(target_rates)
    admixture_coefficients = np.zeros((P, P, duration.size))
    for i in range(P): admixture_coefficients[i, i] = 1.0
    model = PairCoalescenceRateModel(P) if pairs_only else TrioCoalescenceRateModel(P)
    expected_rates = model.forward(demographic_parameters, admixture_coefficients, duration)
    gradient = (np.log(target_rates) - np.log(expected_rates)) * weights ** 2  # wrt log(fitted)
    gradient /= expected_rates
    gradient_parameters, *_ = model.backward(gradient)

    gradient *= expected_rates  # to logspace
    gradient_parameters *= demographic_parameters  # to logspace

    return expected_rates, gradient, gradient_parameters


def draw_gradient_arrows(axs, x, y, length, linewidth=1, alpha=1.0, width=0.2, height=0.02, color="black"):
    for xx, yy, ln in zip(x, y, length):
        yt = 10 ** (np.log10(yy) + ln)
        ytm = 10 ** (np.log10(yy) + ln - height)
        axs.plot([xx, xx], [yy, yt], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")
        if yt > yy:
            axs.plot([xx - width, xx], [ytm, yt], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")
            axs.plot([xx + width, xx], [ytm, yt], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")
        else:
            axs.plot([xx - width, xx], [yt, ytm], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")
            axs.plot([xx + width, xx], [yt, ytm], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")


def optimize_island_model(target, weights, duration, starting_value, lower_bound, upper_bound, pairs_only=True, ftol_rel=1e-6, maxevals=0):
    assert np.logical_and(np.all(starting_value >= lower_bound), np.all(starting_value <= upper_bound))

    num_populations = starting_value.shape[0]
    decoder = PairCoalescenceRateModel(num_populations) if pairs_only \
        else TrioCoalescenceRateModel(num_populations)
    mapping = np.arange(starting_value.size).reshape(starting_value.shape)

    admixture = np.zeros(starting_value.shape)
    for i in range(admixture.shape[-1]): admixture[:, :, i] = np.eye(*admixture[:, :, i].shape)

    # for animation
    sampling_interval = list(np.concatenate([
        np.arange(20),
        np.arange(20, 100, 5),
        np.arange(100, 1000, 50),
        np.arange(1000, 10000, 200),
    ]))

    opt_trajectory = []
    loss_trajectory = []

    def objective(par, grad):
        demography = np.exp(par[mapping])
        fitted = decoder.forward(demography, admixture, duration)
        resid = (target - fitted) * weights
        if grad.size:
            d_demography, *_ = decoder.backward(resid * weights) 
            d_par = np.bincount(mapping.flatten(), weights=d_demography.flatten())
            grad[:] = d_par * np.exp(par)  # logspace
        loglik = -0.5 * np.sum(resid ** 2)
        if len(sampling_interval) > 0 and len(loss_trajectory) == sampling_interval[0]:
            itt = sampling_interval.pop(0)
            opt_trajectory.append((fitted, demography, itt))
        loss_trajectory.append(loglik)
        print(f"{len(loss_trajectory)} loglik {-0.5 * np.sum(resid ** 2)}, grad norm {np.linalg.norm(grad)}")
        return loglik

    lower_bound = np.log(lower_bound).flatten()
    upper_bound = np.log(upper_bound).flatten()
    starting_value = np.log(starting_value).flatten()

    # initialize trajectory with starting state
    objective(starting_value, np.zeros(starting_value.size))
    print(f"Initial loss: {loss_trajectory[0]}")

    optimizer = nlopt.opt(nlopt.LD_LBFGS, starting_value.size)
    optimizer.set_max_objective(objective)
    optimizer.set_lower_bounds(lower_bound)
    optimizer.set_upper_bounds(upper_bound)
    optimizer.set_maxeval(int(maxevals))
    optimizer.set_vector_storage(50)
    optimizer.set_ftol_rel(ftol_rel)
    parameters = optimizer.optimize(starting_value)
    convergence = optimizer.last_optimize_result()
    loglik = optimizer.last_optimum_value()

    demography = np.exp(parameters[mapping])
    fitted = decoder.forward(demography, admixture, duration)

    return demography, fitted, opt_trajectory


def initial_values(time_grid, migr=1e-5, min_migr=1e-10, max_migr=1e-2, ne=1e4, min_ne=1e2, max_ne=1e6):
    _, _, params = rates_and_demography(
        time_grid, 
        intercept=[10 ** (5), 10 ** (4.5)], 
        amplitude=[0.0, 0.0], 
        #pulse_on=[10 ** (-5), 10 ** (-6)], 
        #pulse_off=[10 ** (-5), 10 ** (-6)], 
        pulse_on=[10 ** (-6), 10 ** (-5)], 
        pulse_off=[10 ** (-6), 10 ** (-5)], 
        pairs_only=True,
    )
    st = params.copy()
    lb = params.copy()
    ub = params.copy()
    lb[:] = min_migr
    ub[:] = max_migr
    for i in range(ub.shape[0]):
        lb[i,i] = min_ne
        ub[i,i] = max_ne
    return st, lb, ub
