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
output_dir = "../../fig/sim-singer/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
force_overwrite = True #False
pairs_only = True
grid_size = 50
trim_to = 1e8
min_std = 1e-16

log_grid = np.linspace(2, 5, grid_size)
time_grid = 10 ** log_grid
time_grid[0] = 0

x_min = 1e2
x_max = 1e5
y_min_ne = 4e3
y_max_ne = 1e5
y_min_mi = 5e-7
y_max_mi = 2e-4
y_min_ra = 5e-8 if pairs_only else 1e-11
y_max_ra = 2e-4 if pairs_only else 2e-4

#--------- get rates, etc.

itt = 75
ts_list = [
    #tskit.load(f"/sietch_colab/natep/trio-coal/sims/osclog-alt/singer-snakemake/results/osclogalt_{i}/trees/osclogalt_{i}.{itt}.trees")
    #tskit.load(f"/sietch_colab/natep/trio-coal/sims/osc-log/singer-snakemake/results/osclog_{i}/trees/osclog_{i}.{itt}.trees")
    fetch(i, itt) for i in range(1, 6)
]

rate_store = f"tmp.pair_rates.{itt}.p"
if not os.path.exists(rate_store) or force_overwrite:
    emp_rates, std_rates = calculate_rates(ts_list, time_grid, pairs_only=pairs_only)
    assert np.all(emp_rates >= 0)
    assert np.all(std_rates >= 0)
    duration, rates, params = rates_and_demography(log_grid, pairs_only=pairs_only)
    pickle.dump((duration, rates, params, emp_rates, std_rates), open(rate_store, "wb"))
else:
    duration, rates, params, emp_rates, std_rates = pickle.load(open(rate_store, "rb"))

# ----------

fig = plt.figure(figsize=(8, 4))
fig.supxlabel("Generations in past")

ne_ax = plt.subplot2grid((2, 2), (0, 0))
ne_ax.set_ylim(y_min_ne, y_max_ne)
ne_ax.set_xlim(x_min, x_max)
ne_ax.set_yscale('log')
ne_ax.set_xscale('log')

mi_ax = plt.subplot2grid((2, 2), (1, 0))
mi_ax.set_ylim(y_min_mi, y_max_mi)
mi_ax.set_xlim(x_min, x_max)
mi_ax.set_yscale('log')
mi_ax.set_xscale('log')

ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ra_ax.set_ylim(y_min_ra, y_max_ra)
ra_ax.set_xlim(x_min, x_max)
ra_ax.set_yscale('log')
ra_ax.set_xscale('log')

plot_ne_step(ne_ax, duration, params)
plot_migr_step(mi_ax, duration, params)
plot_rates_step(ra_ax, duration, rates, pairs_only=pairs_only, line_kwargs={}, label_suffix=" from true")
plot_rates_point(ra_ax, duration, emp_rates, pairs_only=pairs_only, point_kwargs={"s" : 3}, label_suffix=" from emp", make_legend=False)
fig.tight_layout()
plt.savefig(output_dir + "test_pair_true.png")


# ----

fit_store = f"tmp.pair_fit.{itt}.p"
if True: #not os.path.exists(fit_store):
    weights = 1 / std_rates
    weights[std_rates == 0] = 0
    emp_params_fit, emp_rates_fit, emp_opt_traj = optimize_island_model(emp_rates, weights, duration, *initial_values(params, min_ne=1e3, max_ne=1e5), pairs_only=pairs_only, ftol_rel=1e-5, penalty=1.0)
    pickle.dump((emp_params_fit, emp_rates_fit, emp_opt_traj), open(fit_store, "wb"))
else:
    emp_params_fit, emp_rates_fit, emp_opt_traj = pickle.load(open(fit_store, "rb"))

for art in list(mi_ax.lines): art.remove()
for art in list(ne_ax.lines): art.remove()
plot_ne_step(ne_ax, duration, emp_params_fit)
plot_migr_step(mi_ax, duration, emp_params_fit)
plt.savefig(output_dir + "test_pair_infr.png")
