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
output_dir = "../../fig/pair-optim/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
force_overwrite = True #False
pairs_only = True
grid_size = 50
trim_to = 1e7


#--------- get rates, etc.

time_grid = np.linspace(0, 5e4, grid_size)
ts = tskit.load("/sietch_colab/natep/trio-coal/sims/osc/oscillating_1.ts")
ts = ts.keep_intervals([[0, trim_to]]).trim()

rate_store = "tmp.pair_rates.p"
if not os.path.exists(rate_store) or force_overwrite:
    emp_rates, std_rates = calculate_rates(ts, time_grid, pairs_only=pairs_only)
    duration, rates, params = rates_and_demography(time_grid, pairs_only=pairs_only)
    pickle.dump((duration, rates, params, emp_rates, std_rates), open(rate_store, "wb"))
else:
    duration, rates, params, emp_rates, std_rates = pickle.load(open(rate_store, "rb"))

traj_store = "tmp.pair_fit.p"
if not os.path.exists(traj_store) or force_overwrite:
    emp_params_fit, emp_rates_fit, emp_opt_traj = optimize_island_model(emp_rates, 1./std_rates, duration, *initial_values(params), pairs_only=pairs_only)
    params_fit, rates_fit, opt_traj = optimize_island_model(rates, 1./std_rates, duration, *initial_values(params), pairs_only=pairs_only, ftol_rel=1e-4)
    pickle.dump((params_fit, rates_fit, opt_traj, emp_params_fit, emp_rates_fit, emp_opt_traj), open(traj_store, "wb"))
else:
    params_fit, rates_fit, opt_traj, emp_params_fit, emp_rates_fit, emp_opt_traj = pickle.load(open(traj_store, "rb"))


# --- look at optimized model

plot_model_fit(duration, params_fit, rates_fit, path=output_dir + "optim-discr-pair-0.png", pairs_only=pairs_only)
plot_model_fit(duration, emp_params_fit, emp_rates_fit, path=output_dir + "optim-discr-pair-1.png", pairs_only=pairs_only)

assert False

# --- make animation of optimization process
def make_anim(duration, target, optimization_traj, path, pairs_only=True):
    start = np.cumsum(np.append(0, duration))[:-1]
    x_min = 0
    x_max = 50
    y_min_ne = 8e3
    y_max_ne = 8e5
    y_min_mi = 1e-6
    y_max_mi = 1e-3
    y_min_ra = 1e-9 if pairs_only else 1e-13
    y_max_ra = 1e-1 if pairs_only else 1e-3
    rate_names = ["(A,A)", "(A,B)", "(B,B)"] if pairs_only else \
        ["((A,A),A)", "((A,A),B)", "((A,B),A)", "((A,B),B)", "((B,B),A)", "((B,B),B)"]
    
    fig = plt.figure(figsize=(8, 4))
    
    ne_ln = {}
    mi_ln = {}
    ra_ln = {}
    ne_ax = plt.subplot2grid((2, 2), (0, 0))
    ne_ln["A"], *_ = ne_ax.plot([], [], label=r"$N_{A}$", color="dodgerblue")
    ne_ln["B"], *_ = ne_ax.plot([], [], label=r"$N_{B}$", color="firebrick")
    ne_ax.set_ylim(y_min_ne, y_max_ne)
    ne_ax.set_xlim(x_min, x_max)
    ne_ax.set_yscale('log')
    ne_ax.set_ylabel("Haploid $N_e$")
    ne_ax.legend(ncol=2, loc='upper left')
    
    mi_ax = plt.subplot2grid((2, 2), (1, 0))
    mi_ln["AB"], *_ = mi_ax.plot([], [], label=r"$M_{A \rightarrow B}$", color="dodgerblue", linestyle='dashed')
    mi_ln["BA"], *_ = mi_ax.plot([], [], label=r"$M_{B \rightarrow A}$", color="firebrick", linestyle='dashed')
    mi_ax.set_yscale('log')
    mi_ax.set_ylim(y_min_mi, y_max_mi)
    mi_ax.set_xlim(x_min, x_max)
    mi_ax.set_ylabel("Migration rate")
    mi_ax.legend(ncol=1, loc='upper left')
    
    ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ra_ln = {}
    for i, label in enumerate(rate_names):
        ra_ln[label], *_ = ra_ax.plot([], [], label=label)
    ra_ta = {}
    for i, label in enumerate(rate_names):
        ra_ta[label], *_ = ra_ax.plot(start / 1e3, target[i], label=label, linestyle='dashed')
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

    num_frames = len(optimization_traj)

    def update(frame):
        rates, params = optimization_traj[frame]
        ne_ln["A"].set_data(start / 1e3, params[0,0])
        ne_ln["B"].set_data(start / 1e3, params[1,1])
        mi_ln["AB"].set_data(start / 1e3, params[0,1])
        mi_ln["BA"].set_data(start / 1e3, params[1,0])
        for i, label in enumerate(rate_names):
            ra_ln[label].set_data(start / 1e3, rates[i])

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, repeat=True, frames=num_frames, interval=100)
    ani.save(path, writer="imagemagick")

#make_anim(duration, rates, opt_traj, path="/home/natep/public_html/trio-pres/tmp/test_pair_demog_discr_optimize.gif", pairs_only=True)
