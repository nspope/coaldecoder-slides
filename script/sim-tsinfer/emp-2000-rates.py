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
output_dir = "../../fig/sim-tsinfer/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
force_overwrite = False
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
y_max_ra = 5e-4 if pairs_only else 3e-4

#--------- get rates, etc.

#rate_store = f"tmp.2000_rates.p"
rate_store = f"tmp.himu_rates.p"
if not os.path.exists(rate_store) or force_overwrite:
    ts_list = [
        #fetch_big(i, j) for i in range(5) for j in range(1, 6)
        fetch_himu(i, j) for i in range(5) for j in range(1, 6)
    ]
    emp_rates, std_rates, _ = calculate_rates(ts_list, time_grid, pairs_only=pairs_only)
    emp_trio_rates, std_trio_rates, trio_rates_dbg = calculate_rates(ts_list, time_grid, pairs_only=False)
    duration, rates, params = rates_and_demography(log_grid, pairs_only=pairs_only)
    _, trio_rates, _ = rates_and_demography(log_grid, pairs_only=False)
    pickle.dump((duration, rates, trio_rates, params, emp_rates, std_rates, emp_trio_rates, std_trio_rates), open(rate_store, "wb"))
    pickle.dump(trio_rates_dbg, open("debug-2000.p", "wb"))
else:
    (duration, rates, trio_rates, params, emp_rates, std_rates, emp_trio_rates, std_trio_rates) = pickle.load(open(rate_store, "rb"))

assert np.all(emp_rates >= 0)
assert np.all(std_rates >= 0)
assert np.all(emp_trio_rates[6:] >= 0)
assert np.all(std_trio_rates[6:] >= 0)

# ----------

fig = plt.figure(figsize=(8, 4))
fig.supxlabel("Generations in past")

ne_ax = plt.subplot2grid((2, 2), (0, 0))
ne_ax.set_ylim(y_min_ne, y_max_ne)
ne_ax.set_xlim(x_min, x_max)
ne_ax.set_yscale('log')
ne_ax.set_xscale('log')
ne_ax.set_ylabel("Haploid $N_e$")

mi_ax = plt.subplot2grid((2, 2), (1, 0))
mi_ax.set_ylim(y_min_mi, y_max_mi)
mi_ax.set_xlim(x_min, x_max)
mi_ax.set_yscale('log')
mi_ax.set_xscale('log')
mi_ax.set_ylabel("Migration rate")

ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ra_ax.set_ylim(y_min_ra, y_max_ra)
ra_ax.set_xlim(x_min, x_max)
ra_ax.set_yscale('log')
ra_ax.set_xscale('log')

plot_ne_step(ne_ax, duration, params)
plot_migr_step(mi_ax, duration, params)
plot_rates_step(ra_ax, duration, rates, pairs_only=pairs_only, line_kwargs={}, label_suffix=" from true")
ra_label = ra_ax.text(0.02, 0.98, "Expected rates", ha='left', va='top', transform=ra_ax.transAxes)
ne_label = ne_ax.text(0.98, 0.96, "Generative model", ha='right', va='top', transform=ne_ax.transAxes)
#fig.suptitle("tsinfer 0.3.3 + tsdate 0.1.6dev\n10 Mb, 400 diploids, $\mu/r = 2$")
fig.tight_layout()

# inferred pair rates
for art in list(ra_ax.lines): art.remove()
ra_label.remove()
ra_label = ra_ax.text(0.02, 0.98, "Inferred rates (tsinfer+tsdate, 400 dip)", ha='left', va='top', transform=ra_ax.transAxes)
points = plot_rates_point(ra_ax, duration, emp_rates, pairs_only=pairs_only, point_kwargs={"s" : 3}, label_suffix=" from ecdf", make_legend=True)
plot_rates_step(ra_ax, duration, rates, pairs_only=pairs_only, line_kwargs={"alpha" : 0.2}, label_suffix=" from true", make_legend=False)
plt.savefig(output_dir + "emp-pair-2000-0.png")

#ra_rect = add_highlight(ra_ax, highlight=[3e4, 1e5])
#mi_rect = add_highlight(mi_ax, highlight=[3e4, 1e5])
#ne_rect = add_highlight(ne_ax, highlight=[3e4, 1e5])
#emoji = ra_ax.text(0.9, 0.8, "!?", size=14, transform=ra_ax.transAxes)
#plt.savefig(output_dir + "emp-pair-2000-3.png")
#emoji.remove()

# inferred trio rates
for art in list(ra_ax.lines): art.remove()
for art in points: art.remove()
#ra_rect.remove()
#ne_rect.remove()
#mi_rect.remove()
points = plot_rates_point(ra_ax, duration, emp_trio_rates[6:], pairs_only=False, point_kwargs={"s" : 3}, label_suffix=" from ecdf", make_legend=True)
plot_rates_step(ra_ax, duration, trio_rates[6:], pairs_only=False, line_kwargs={"alpha" : 0.2}, label_suffix=" from true", make_legend=False)
ra_rect = add_highlight(ra_ax, highlight=[3e4, 1e5])
mi_rect = add_highlight(mi_ax, highlight=[3e4, 1e5])
ne_rect = add_highlight(ne_ax, highlight=[3e4, 1e5])
ra_ax.set_ylim(1e-10, 1e-3)
plt.savefig(output_dir + "emp-pair-2000-1.png")
assert False

# ----

fit_store = f"tmp.200_fit.p"
if not os.path.exists(fit_store) or force_overwrite:
    weights = 1 / std_rates
    weights[std_rates == 0] = 0
    emp_params_fit, emp_rates_fit, emp_opt_traj = optimize_island_model(emp_rates, weights, duration, *initial_values(params), pairs_only=pairs_only, ftol_rel=1e-4, penalty=5.0) #for mu=2e-8

    emp_trio_rates[:6] = 0
    std_trio_rates[:6] = 0
    weights = 1 / std_trio_rates
    weights[std_trio_rates == 0] = 0
    weights[9] = 0 # ignore (A,B)B
    weights[8] = 0 # ignore (A,B)B
    emp_trio_params_fit, emp_trio_rates_fit, emp_trio_opt_traj = optimize_island_model(emp_trio_rates, weights, duration, *initial_values(params), pairs_only=False, ftol_rel=1e-4, penalty=30.0) #for mu=2e-8

    pickle.dump((emp_params_fit, emp_rates_fit, emp_opt_traj, emp_trio_params_fit, emp_trio_rates_fit, emp_trio_opt_traj), open(fit_store, "wb"))
else:
    (emp_params_fit, 
     emp_rates_fit, 
     emp_opt_traj, 
     emp_trio_params_fit, 
     emp_trio_rates_fit, 
     emp_trio_opt_traj) = pickle.load(open(fit_store, "rb"))

for art in list(mi_ax.lines): art.remove()
for art in list(ne_ax.lines): art.remove()
plot_ne_step(ne_ax, duration, emp_trio_params_fit)
mi_ax.get_legend().remove()
plot_migr_step(mi_ax, duration, emp_trio_params_fit, legend_loc='lower left', legend_bbox=[0.3, 0.01])
ne_label.remove()
ne_label = ne_ax.text(0.98, 0.96, "Fitted model", ha='right', va='top', transform=ne_ax.transAxes)
plt.savefig(output_dir + "emp-pair-2000-5.png")
