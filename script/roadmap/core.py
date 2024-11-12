import numpy as np
import tskit
import os
import pickle
import nlopt
import matplotlib
import msprime

import matplotlib.pyplot as plt

def fetch(i, j):
    fname = f"/sietch_colab/natep/trio-coal/sims/osclogalt-big-regmu/infer_osclogaltbig_{j}.{i}.trees"
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

    return np.diff(time_grid), demographic_parameters[:2, :2], admixture_coefficients[:2, :2]


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


def get_pdf_msprime_unfitted():
    log_grid = np.linspace(2, 5, 1001)
    time_step, params, admix = rates_and_demography(log_grid, amplitude=[0, 0], intercept=[1e4, 5e4], pulse_off=[1e-5, 1e-5], pulse_on=[1e-5, 1e-5])
    params[:] = initial_values(params)[0]
    params[0,0] = 1e4
    params[1,1] = 5e4
    demog = to_msprime(params, admix, time_step, ["A", "B"])
    time_step = np.linspace(0, 6e4, 1001)
    pdf = []
    for lineages in [{"A" : 2}, {"A" : 1, "B" : 1}, {"B" : 2}]:
        traj = demog.debug().coalescence_rate_trajectory(lineages=lineages, steps=time_step)
        haz = traj[0]
        cdf = 1 - traj[1]
        mass = np.diff(cdf) / np.diff(time_step)
        mid = time_step[1:]/2 + time_step[:-1]/2
        pdf.append((mid, mass, haz))
    return pdf


def get_pdf_msprime_fitted():
    log_grid = np.linspace(2, 5, 1001)
    time_step, params, admix = rates_and_demography(log_grid)
    demog = to_msprime(params, admix, time_step, ["A", "B"])
    time_step = np.linspace(0, 6e4, 1001)
    pdf = []
    for lineages in [{"A" : 2}, {"A" : 1, "B" : 1}, {"B" : 2}]:
        traj = demog.debug().coalescence_rate_trajectory(lineages=lineages, steps=time_step)
        haz = traj[0]
        cdf = 1 - traj[1]
        mass = np.diff(cdf) / np.diff(time_step)
        mid = time_step[1:]/2 + time_step[:-1]/2
        pdf.append((mid, mass, haz))
    return pdf
