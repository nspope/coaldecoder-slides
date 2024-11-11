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
duration, true_rates, true_params = rates_and_demography(time_grid, pulse_on=[1e-4, 1e-3], pairs_only=pairs_only)
duration, rates, params = rates_and_demography(time_grid, intercept=[10 ** (5), 10 ** (4.5)], amplitude=[0.0, 0.0], pulse_on=[10 ** (-6), 10 ** (-5)], pulse_off=[10 ** (-6), 10 ** (-5)], pairs_only=pairs_only)
rate_check, grad_rates, grad_params = calculate_gradient(true_rates, params, duration)

x_min = 0
x_max = 50
y_min_ne = 8e3
y_max_ne = 8e5
y_min_mi = 1e-7
y_max_mi = 3e-3
y_min_ra = 5e-8 if pairs_only else 1e-13
y_max_ra = 2e-4 if pairs_only else 1e-3

# --- introduce model

fig = plt.figure(figsize=(8, 4))
fig.supxlabel("Thousands of generations in past")

ne_ax = plt.subplot2grid((2, 2), (0, 0))
ne_ax.set_ylim(y_min_ne, y_max_ne)
ne_ax.set_xlim(x_min, x_max)
plot_ne(ne_ax, duration, true_params)

mi_ax = plt.subplot2grid((2, 2), (1, 0))
plot_migr(mi_ax, duration, true_params)
mi_ax.set_ylim(y_min_mi, y_max_mi)
mi_ax.set_xlim(x_min, x_max)

ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
plot_rates(ra_ax, duration, true_rates, pairs_only=True)
ra_ax.set_ylim(y_min_ra, y_max_ra)
ra_ax.set_xlim(x_min, x_max)

fig.tight_layout()
label_ra = ra_ax.text(0.02, 0.98, "``Observed'' rates", ha="left", va="top", transform=ra_ax.transAxes)
label_ne = ne_ax.text(0.98, 0.96, "Generative model", ha="right", va="top", transform=ne_ax.transAxes)
plt.savefig(output_dir + "pair-optim-0.png")

for art in list(ra_ax.lines): art.remove()
for art in list(ne_ax.lines): art.remove()
for art in list(mi_ax.lines): art.remove()
plot_ne(ne_ax, duration, params)
plot_migr(mi_ax, duration, params)
plot_rates(ra_ax, duration, true_rates, pairs_only=True, line_kwargs={"alpha" : 0.2}, draw_legend=False)
plot_rates(ra_ax, duration, rates, pairs_only=True, draw_legend=False)
label_ra.remove()
label_ne.remove()
ne_ax.set_ylabel("Fitted haploid $N_e$")
mi_ax.set_ylabel("Fitted migration rate")
label_ra = ra_ax.text(0.02, 0.98, "Candidate rates", ha="left", va="top", transform=ra_ax.transAxes)
label_ne = ne_ax.text(0.98, 0.96, "Candidate model", ha="right", va="top", transform=ne_ax.transAxes)
plt.savefig(output_dir + "pair-optim-1.png")

# --- with gradient

rates_arrow_length = 0.3
grad_rates /= np.abs(grad_rates).max() 
grad_rates *= rates_arrow_length
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[i % len(colors)] for i in range(rates.shape[0])]
gap = 10
for i in range(rates.shape[0]):
    draw_gradient_arrows(ra_ax, time_grid[::gap] / 1e3, rates[i][::gap], grad_rates[i][::gap], color=colors[i])
label_ra.remove()
label_ra = ra_ax.text(0.02, 0.98, "Differentiate wrt loss", ha="left", va="top", transform=ra_ax.transAxes)

plt.savefig(output_dir + "pair-optim-2.png")

# --- backpropagate

params_arrow_length = 0.3
grad_params_ne = np.stack([grad_params[0, 0], grad_params[1, 1]])
grad_params_ne /= np.abs(grad_params_ne).max() 
grad_params_ne *= params_arrow_length
grad_params_mi = np.stack([grad_params[0, 1], grad_params[1, 0]])
grad_params_mi /= np.abs(grad_params_mi).max() 
grad_params_mi *= params_arrow_length
gap = 10
draw_gradient_arrows(ne_ax, time_grid[::gap] / 1e3, params[0, 0][::gap], grad_params_ne[0][::gap], color="dodgerblue")
draw_gradient_arrows(ne_ax, time_grid[::gap] / 1e3, params[1, 1][::gap], grad_params_ne[1][::gap], color="firebrick")
draw_gradient_arrows(mi_ax, time_grid[::gap] / 1e3, params[0, 1][::gap], grad_params_mi[0][::gap], color="dodgerblue")
draw_gradient_arrows(mi_ax, time_grid[::gap] / 1e3, params[1, 0][::gap], grad_params_mi[1][::gap], color="firebrick")
label_ne.remove()
label_ne = ne_ax.text(0.98, 0.96, "Backpropagate", ha="right", va="top", transform=ne_ax.transAxes)

plt.savefig(output_dir + "pair-optim-3.png")



