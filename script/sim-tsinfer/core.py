import numpy as np
import tskit
import os
import pickle
import nlopt
import matplotlib
import msprime

from coaldecoder import TrioCoalescenceRateModel, PairCoalescenceRateModel
from coaldecoder import TrioCoalescenceRatesPoly, PairCoalescenceRates

import matplotlib.pyplot as plt

WHICH_MODEL = "osclogalt"

def fetch(i, j):
    #fname = f"/sietch_colab/natep/trio-coal/sims/osclog-altbig/osclogaltbig_{j}.{i}.trees"
    fname = f"/sietch_colab/natep/trio-coal/sims/osclogalt-big-regmu/infer_osclogaltbig_{j}.{i}.trees"
    #fname = f"/sietch_colab/natep/trio-coal/sims/osclogalt-huge/infer_osclogaltbig_{j}.{i}.trees"
    print(fname)
    return tskit.load(fname)

def plogis(x):
    return 1. / (1. + np.exp(-x))

def rates_and_demography(
        log_grid,
        intercept=[2e4, 2e4], 
        phase=[0, 1.0], 
        frequency=[0.50, 0.50], 
        amplitude=[15e3, 15e3], 
        pulse_decay=[4, -4], 
        pulse_mid=[3.5, 3.5], 
        pulse_on=[-4, -4], 
        pulse_off=[-6, -6], 
        pairs_only=True,
    ):
    """
    Pair rates for two oscillating populations with pulse migration
    """

    time_grid = 10 ** log_grid
    time_grid[0] = 0.0
    grid_size = time_grid.size

    log_start = log_grid[:-1]
    epoch_start = time_grid[:-1]

    demographic_parameters = np.zeros((3, 3, grid_size - 1))
    demographic_parameters[0,0] = intercept[0] + amplitude[0] * np.cos(2 * np.pi * (log_start + phase[0]) * frequency[0])
    demographic_parameters[0,1] = 10 ** (pulse_off[0] + plogis(pulse_decay[0] * (log_start - pulse_mid[0])) * (pulse_on[0] - pulse_off[0]))
    demographic_parameters[1,0] = 10 ** (pulse_off[1] + plogis(pulse_decay[1] * (log_start - pulse_mid[1])) * (pulse_on[1] - pulse_off[1]))
    demographic_parameters[1,1] = intercept[1] + amplitude[1] * np.cos(2 * np.pi * (log_start + phase[1]) * frequency[1])
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



def to_msprime(demographic_parameters, admixture_coefficients, time_step, population_names):
    assert demographic_parameters.shape == admixture_coefficients.shape
    assert len(population_names) == demographic_parameters.shape[0] == demographic_parameters.shape[1]
    assert len(time_step) == demographic_parameters.shape[2]

    demography = msprime.Demography()
    for i, p in enumerate(population_names):
        demography.add_population(initial_size=np.inf, name=p)

    start_time = np.cumsum(np.append(0, time_step))
    demographic_parameters = demographic_parameters.transpose(2, 0, 1)
    admixture_coefficients = admixture_coefficients.transpose(2, 0, 1)
    for M, A, t in zip(demographic_parameters, admixture_coefficients, start_time):
        for i, p in enumerate(population_names):
            for j, q in enumerate(population_names):
                if i != j and A[j, i] > 0:
                    demography.add_mass_migration(time=t, source=p, dest=q, proportion=A[j, i])

        for i, p in enumerate(population_names):
            for j, q in enumerate(population_names):
                if i == j:
                    demography.add_population_parameters_change(time=t, initial_size=M[i, i] / 2, population=p)
                else:
                    demography.add_migration_rate_change(time=t, rate=M[i, j], source=p, dest=q)

    return demography


def difference_operator(dim, order=1):
    diff = np.eye(dim, dim)
    for i in range(order):
        d = -np.eye(diff.shape[0] - 1, diff.shape[0])
        for j in range(d.shape[0]):
            d[j, j + 1] = 1
        diff = d @ diff
    return diff

def smoothing_penalty(parameters, penalty, operator):
    log_pen = 0
    d_parameters = np.zeros_like(parameters)
    for i in range(parameters.shape[0]):
        for j in range(parameters.shape[1]):
            diff_ij = operator @ np.log10(parameters[i, j])
            log_pen += np.sum(-0.5 * (diff_ij * penalty[i, j]) ** 2)
            diff_ij = -1 * diff_ij * penalty[i, j] ** 2
            d_parameters[i, j] = operator.T @ diff_ij
            d_parameters[i, j] /= parameters[i, j] * np.log(10)
    return log_pen, d_parameters


def test_smoothing_penalty():
    import numdifftools as nd
    np.random.seed(1)
    operator = difference_operator(4)
    penalty = np.arange(3 * 3, dtype=float).reshape(3, 3)
    parameters = np.zeros((3, 3, 4))
    check_pen = 0
    for i in range(3):
        for j in range(3):
            parameters[i, j] = np.random.uniform(1e2, 1e3, size=parameters.shape[-1])
            check_pen += np.sum(-0.5 * (np.diff(np.log10(parameters[i, j])) * penalty[i, j]) ** 2)
    test_pen, test_grad = smoothing_penalty(parameters, penalty, operator)
    def wrapper(x):
        p = parameters.copy()
        p[:, :, :] = x.reshape(p.shape)
        return smoothing_penalty(p, penalty, operator)[0]
    check_grad = nd.Gradient(wrapper, n=1, step=1e-4)(parameters.flatten()).reshape(parameters.shape)
    assert np.isclose(test_pen, check_pen)
    np.testing.assert_allclose(test_grad, check_grad, rtol=1e-4)

#test_smoothing_penalty()


def optimize_island_model(target, weights, duration, starting_value, lower_bound, upper_bound, pairs_only=True, ftol_rel=1e-6, maxevals=0, penalty=0.0):
    assert np.logical_and(np.all(starting_value >= lower_bound), np.all(starting_value <= upper_bound))

    num_populations = starting_value.shape[0]
    decoder = PairCoalescenceRateModel(num_populations) if pairs_only \
        else TrioCoalescenceRateModel(num_populations)
    mapping = np.arange(starting_value.size).reshape(starting_value.shape)

    admixture = np.zeros(starting_value.shape)
    for i in range(admixture.shape[-1]): admixture[:, :, i] = np.eye(*admixture[:, :, i].shape)

    # for animation
    sampling_interval = 10
    max_samples = 100

    # smoothing penalty
    penalty = np.full((num_populations, num_populations), penalty)
    np.fill_diagonal(penalty, 0.0)
    operator = difference_operator(starting_value.shape[-1])

    opt_trajectory = []
    loss_trajectory = []

    def objective(par, grad):
        demography = np.exp(par[mapping])
        logpen, d_penalty = smoothing_penalty(demography, penalty, operator)
        fitted = decoder.forward(demography, admixture, duration)
        resid = (target - fitted) * weights
        if grad.size:
            d_demography, *_ = decoder.backward(resid * weights) 
            d_demography += d_penalty # penalty
            d_par = np.bincount(mapping.flatten(), weights=d_demography.flatten())
            grad[:] = d_par * np.exp(par)  # logspace
        loglik = -0.5 * np.sum(resid ** 2)
        loss_trajectory.append(loglik)
        if len(loss_trajectory) % sampling_interval == 0 and len(opt_trajectory) < max_samples:
            opt_trajectory.append((fitted, demography))
        #if len(loss_trajectory) % sampling_interval == 0:
        #    print(f"{len(loss_trajectory)} loglik {-0.5 * np.sum(resid ** 2)}, grad norm {np.linalg.norm(grad)}")
        print(f"{len(loss_trajectory)} loglik {loglik}, penalty {logpen}, grad norm {np.linalg.norm(grad)}")
        return loglik + logpen

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


def initial_values(params, migr=1e-5, min_migr=1e-10, max_migr=1e-2, ne=1e4, min_ne=1e2, max_ne=1e6):
    st = params.copy()
    lb = params.copy()
    ub = params.copy()
    st[:] = migr
    lb[:] = min_migr
    ub[:] = max_migr
    for i in range(ub.shape[0]):
        st[i,i] = ne
        lb[i,i] = min_ne
        ub[i,i] = max_ne
    return st, lb, ub


def calculate_rates(ts_list, time_grid, num_blocks=10, random_seed=1, pairs_only=True):
    ts = ts_list[0]

    population_map = {0 : "A" , 1 : "B"}
    sample_population = np.full(ts.num_samples, "")
    for i in range(ts.num_samples):
        sample_population[i] = population_map[ts.nodes_population[i]]
    population_names = np.unique(sample_population)
    sample_sets = []
    for p in population_names:
        sample_sets.append(np.flatnonzero(sample_population == p))
    
    rates_calculator = None
    calculator = PairCoalescenceRates if pairs_only else TrioCoalescenceRatesPoly
    for ts in ts_list:
        #windows = np.linspace(0, ts.sequence_length, num_blocks + 1)
        windows = np.linspace(0, ts.sequence_length, 2)
        if rates_calculator is None:
            rates_calculator = calculator(ts, sample_sets, time_grid, sample_set_names=population_names, bootstrap_blocks=windows, check_binary=False)
        else:
            rates_calculator.join(calculator(ts, sample_sets, time_grid, sample_set_names=population_names, bootstrap_blocks=windows, check_binary=False))
        print("Sequence length:", rates_calculator.sequence_length)

    emp_rates = rates_calculator.rates()
    std_rates = rates_calculator.std_dev(num_replicates=num_blocks, random_seed=random_seed)

    return emp_rates, std_rates


# ----

def plot_model_fit(duration, params, rates, path, highlight=None, pairs_only=True):
    start = np.cumsum(np.append(0, duration))[:-1]
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    
    fig = plt.figure(figsize=(8, 4))
    if highlight is not None:
        def focal_highlight():
            return matplotlib.patches.Rectangle((highlight[0], 1e-30), highlight[1] - highlight[0], 1e30, fc = 'gray', alpha=0.3)
    
    offset = 0.2
    ne_ax = plt.subplot2grid((2, 2), (0, 0))
    ne_ax.step(start / 1e3, params[0,0], label=r"$N_{A}$", color="dodgerblue")
    ne_ax.step(start / 1e3 + offset, params[1,1], label=r"$N_{B}$", color="firebrick")
    ne_ax.set_ylim(8e3, 8e5)
    ne_ax.set_yscale('log')
    ne_ax.set_ylabel("Haploid $N_e$")
    ne_ax.legend(ncol=2, loc='upper left')
    if highlight is not None:
        ne_ax.add_patch(focal_highlight())
    
    mi_ax = plt.subplot2grid((2, 2), (1, 0))
    mi_ax.step(start / 1e3, params[0,1], label=r"$M_{A \rightarrow B}$", color="dodgerblue")
    mi_ax.step(start / 1e3 + offset, params[1,0], label=r"$M_{B \rightarrow A}$", color="firebrick")
    mi_ax.set_yscale('log')
    mi_ax.set_ylabel("Migration rate")
    mi_ax.legend(ncol=1, loc='upper left')
    if highlight is not None:
        mi_ax.add_patch(focal_highlight())
    
    ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    for i, label in enumerate(rate_names):
        ra_ax.step(start / 1e3 + i * offset, rates[i], label=label)
    ra_ax.set_yscale('log')
    ra_ax.set_ylabel("Pair coalescence rate")
    ra_ax.legend(ncol=1, loc='lower right')
    if highlight is not None:
        ra_ax.add_patch(focal_highlight())
    
    fig.supxlabel("Thousands of generations in past")
    fig.tight_layout()
    plt.savefig(path)
    plt.clf()


# ---


def plot_ne_step(ne_ax, duration, params, line_kwargs={}):
    start = np.cumsum(np.append(0, duration))[:-1]
    ne_ax.step(start, params[0,0], label=r"$N_{A}$", color="dodgerblue", where="post", **line_kwargs)
    ne_ax.step(start, params[1,1], label=r"$N_{B}$", color="firebrick", where="post", **line_kwargs)
    ne_ax.legend(ncol=2, loc='upper left')


def plot_migr_step(mi_ax, duration, params, line_kwargs={}):
    start = np.cumsum(np.append(0, duration))[:-1]
    mi_ax.step(start, params[0,1], label=r"$M_{A \rightarrow B}$", color="dodgerblue", where="post", **line_kwargs)
    mi_ax.step(start, params[1,0], label=r"$M_{B \rightarrow A}$", color="firebrick", where="post", **line_kwargs)
    mi_ax.legend(ncol=1, loc='center left', bbox_to_anchor=(0.01, 0.5))


def plot_rates_step(ra_ax, duration, rates, pairs_only=True, colors=None, line_kwargs={}, make_legend=True, label_suffix=""):
    start = np.cumsum(np.append(0, duration))[:-1]
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    if colors is None:
        colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [colors[i % len(colors)] for i in range(len(rate_names))]
    for i, label in enumerate(rate_names):
        ra_ax.step(start, rates[i], label=label + label_suffix, color=colors[i], where="post", **line_kwargs)
    if pairs_only:
        ra_ax.set_ylabel("Pair coalescence rate")
    else:
        ra_ax.set_ylabel("Trio coalescence rate")
    if make_legend:
        ra_ax.legend(ncol=1, loc='lower right')


def plot_rates_point(ra_ax, duration, rates, pairs_only=True, colors=None, point_kwargs={}, make_legend=True, label_suffix=""):
    start = np.cumsum(np.append(0, duration))[:-1]
    end = np.cumsum(duration)
    mid = start / 2 + end / 2
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    if colors is None:
        colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [colors[i % len(colors)] for i in range(len(rate_names))]
    for i, label in enumerate(rate_names):
        #ra_ax.step(start / 1e3 + offset * i, rates[i], label=label + label_suffix, color=colors[i], where="post")
        pts = ra_ax.scatter(mid, rates[i], label=label + label_suffix, c=colors[i], **point_kwargs)
    if make_legend:
        labs = [nm + label_suffix for nm in rate_names]
        ra_ax.legend(labs, ncol=1, loc='lower right')


def add_highlight(axs, highlight):
    rect = matplotlib.patches.Rectangle((highlight[0], 1e-30), highlight[1] - highlight[0], 1e30, fc = 'gray', alpha=0.3)
    if highlight is not None:
        axs.add_patch(rect)
    return rect
