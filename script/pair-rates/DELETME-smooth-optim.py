import numpy as np
import tskit
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from core import *

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# ----------- #
output_dir = "../../fig/pair-rates/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
force_overwrite = False
pairs_only = True
grid_size = 1000

time_grid = np.linspace(0, 5e4, grid_size)

# --- optimize and save trajectory --- #

st, lb, ub = initial_values(params)
st_rates ... = 
if ...:
    params_fit, rates_fit, opt_traj = optimize_island_model(rates, 1./rates.max(), duration, st, lb, ub, pairs_only=pairs_only, ftol_rel=1e-4)
    pickle.dump((params_fit, rates_fit, opt_traj), open(traj_store, "wb"))

... = optimize_island_model(...)

# ----------- #

x_min = 0
x_max = 50
y_min_ne = 8e3
y_max_ne = 8e5
y_min_mi = 1e-6
y_max_mi = 1e-3
y_min_ra = 5e-8 if pairs_only else 1e-13
y_max_ra = 2e-4 if pairs_only else 1e-3
rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
    ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]

fig = plt.figure(figsize=(8, 4))

ne_ln = {}
mi_ln = {}
ra_ln = {}
ne_ax = plt.subplot2grid((2, 2), (0, 0))
ne_ln["A"], *_ = ne_ax.plot(start / 1e3, st[0, 0], label=r"$N_{A}$", color="dodgerblue")
ne_ln["B"], *_ = ne_ax.plot(start / 1e3, st[1, 1], label=r"$N_{B}$", color="firebrick")
ne_ax.set_ylim(y_min_ne, y_max_ne)
ne_ax.set_xlim(x_min, x_max)
ne_ax.set_yscale('log')
ne_ax.set_ylabel("Haploid $N_e$")
ne_ax.legend(ncol=2, loc='upper left')

mi_ax = plt.subplot2grid((2, 2), (1, 0))
mi_ln["AB"], *_ = mi_ax.plot(start / 1e3, st[0, 1], label=r"$M_{A \rightarrow B}$", color="dodgerblue", linestyle='dashed')
mi_ln["BA"], *_ = mi_ax.plot(start / 1e3, st[1, 0], label=r"$M_{B \rightarrow A}$", color="firebrick", linestyle='dashed')
mi_ax.set_yscale('log')
mi_ax.set_ylim(y_min_mi, y_max_mi)
mi_ax.set_xlim(x_min, x_max)
mi_ax.set_ylabel("Migration rate")
mi_ax.legend(ncol=1, loc='upper left')

ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ra_ln = {}
for i, label in enumerate(rate_names):
    ra_ln[label], *_ = ra_ax.plot(start / 1e3, st_rates[i], label=label)
ra_ax.set_yscale('log')
ra_ax.set_ylim(y_min_ra, y_max_ra)
ra_ax.set_xlim(x_min, x_max)
if pairs_only:
    ra_ax.set_ylabel("Pair coalescence rate")
else:
    ra_ax.set_ylabel("Trio coalescence rate")
ra_ax.legend(ncol=1, loc='lower right')

fig.supxlabel("Thousands of generations in past")
fig.tight_layout()
plt.savefig(output_dir + "smooth-optim-0.png")

matplotlib.rcParams['figure.dpi'] = 100  # reset

# TODO from here

# smooth deformation of parameters
num_frames = 100
assert num_frames % 2 == 0
phase_A = np.linspace(0, 3e4, num_frames)
amplt_B = np.linspace(9e4, -9e4, num_frames // 2)
amplt_B = np.append(amplt_B, amplt_B[::-1])
pulse_B = np.logspace(-6, -3, num_frames // 2)
pulse_B = np.logspace(-6, np.log10(2e-3), num_frames // 2)
pulse_B = np.append(pulse_B, pulse_B[::-1])
mode_A = np.linspace(0, 2e4, num_frames // 2)[::-1]
mode_A = np.append(mode_A, mode_A[::-1])

def update(frame):
    phase = [phase_A[frame], 5e3]
    amplitude = [9e4, amplt_B[frame]]
    pulse_on = [5e-4, pulse_B[frame]]
    pulse_mode = [mode_A[frame], 4e4]
    duration, rates, params = rates_and_demography(
        time_grid, 
        pairs_only=pairs_only, 
        phase=phase, 
        amplitude=amplitude, 
        pulse_on=pulse_on, 
        pulse_mode=pulse_mode,
    )
    start = time_grid[:-1]
    ne_ln["A"].set_data(start / 1e3, params[0,0])
    ne_ln["B"].set_data(start / 1e3, params[1,1])
    mi_ln["AB"].set_data(start / 1e3, params[0,1])
    mi_ln["BA"].set_data(start / 1e3, params[1,0])
    for i, label in enumerate(rate_names):
        ra_ln[label].set_data(start / 1e3, rates[i])

ani = FuncAnimation(fig, update, repeat=True, frames=num_frames, interval=100)
writer = PillowWriter(fps=20)
ani.save(output_dir + "smooth-anim-0.gif", writer=writer, dpi=300)
