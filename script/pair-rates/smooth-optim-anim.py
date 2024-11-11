import numpy as np
import tskit
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt

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
duration, target, true_params = rates_and_demography(time_grid, pulse_on=[2e-4, 1e-3], pairs_only=pairs_only)
weights = np.full_like(target, 1.0 / target.max())
assert np.all(weights > 0)
assert np.all(target > 0)

traj_store = "tmp.pair_fit.p"
if not os.path.exists(traj_store) or force_overwrite:
    params_fit, rates_fit, opt_traj = optimize_island_model(target, weights, duration, *initial_values(time_grid), pairs_only=pairs_only, ftol_rel=1e-6, maxevals=1e4)
    pickle.dump((params_fit, rates_fit, opt_traj), open(traj_store, "wb"))
else:
    params_fit, rates_fit, opt_traj = pickle.load(open(traj_store, "rb"))

# ----------------

x_min = 0
x_max = 50
y_min_ne = 8e3
y_max_ne = 8e5
y_min_mi = 1e-7
y_max_mi = 3e-3
y_min_ra = 5e-8 if pairs_only else 1e-13
y_max_ra = 2e-4 if pairs_only else 1e-3

# --- introduce model

rate_names = ["(A,A)", "(A,B)", "(B,B)"]
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[i % len(colors)] for i in range(len(rate_names))]
start = time_grid[:-1]

fig = plt.figure(figsize=(8, 4))
fig.supxlabel("Thousands of generations in past")

ne_ax = plt.subplot2grid((2, 2), (0, 0))
ne_ax.set_ylim(y_min_ne, y_max_ne)
ne_ax.set_xlim(x_min, x_max)
ne_ax.set_yscale('log')
ne_ax.set_ylabel("Fitted haploid $N_e$")
ne_ln = {}
ne_ln["A"], *_ = ne_ax.plot([], [], label=r"$N_{A}$", color="dodgerblue")
ne_ln["B"], *_ = ne_ax.plot([], [], label=r"$N_{B}$", color="firebrick")
ne_ax.legend(ncol=2, loc='upper left')

mi_ax = plt.subplot2grid((2, 2), (1, 0))
mi_ax.set_yscale('log')
mi_ax.set_ylim(y_min_mi, y_max_mi)
mi_ax.set_xlim(x_min, x_max)
mi_ax.set_ylabel("Fitted migration rate")
mi_ln = {}
mi_ln["AB"], *_ = mi_ax.plot([], [], label=r"$M_{A \rightarrow B}$", color="dodgerblue")
mi_ln["BA"], *_ = mi_ax.plot([], [], label=r"$M_{B \rightarrow A}$", color="firebrick")
mi_ax.legend(ncol=1, loc='upper left')

ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ra_ax.set_yscale('log')
ra_ax.set_ylim(y_min_ra, y_max_ra)
ra_ax.set_xlim(x_min, x_max)
ra_ax.set_ylabel("Pair coalescence rate")
ra_ta = {}
for i, label in enumerate(rate_names):
    ra_ta[label], *_ = ra_ax.plot(start / 1e3, target[i], color=colors[i], label=label)
ra_ax.legend(ncol=1, loc='lower right')
for label in rate_names:
    ra_ta[label].set_alpha(0.2)
ra_ln = {}
for i, label in enumerate(rate_names):
    ra_ln[label], *_ = ra_ax.plot([], [], color=colors[i], label=label)

iteration_label = ra_ax.text(0.01, 0.99, "", ha="left", va="top", transform=ra_ax.transAxes)

fig.tight_layout()

num_frames = len(opt_traj)

def update(frame):
    rates, params, itt = opt_traj[frame]
    ne_ln["A"].set_data(start / 1e3, params[0,0])
    ne_ln["B"].set_data(start / 1e3, params[1,1])
    mi_ln["AB"].set_data(start / 1e3, params[0,1])
    mi_ln["BA"].set_data(start / 1e3, params[1,0])
    for i, label in enumerate(rate_names):
        ra_ln[label].set_data(start / 1e3, rates[i])
    iteration_label.set_text(f"Optimizer step {itt}")

from matplotlib.animation import FuncAnimation, PillowWriter
ani = FuncAnimation(fig, update, repeat=True, frames=num_frames, interval=100)
writer = PillowWriter(fps=20)
ani.save(output_dir + "smooth-optim-anim-0.gif", writer=writer, dpi=300)
