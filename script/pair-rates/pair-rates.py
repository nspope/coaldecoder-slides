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
duration, rates, params = rates_and_demography(time_grid, pairs_only=pairs_only)

x_min = 0
x_max = 50
y_min_ne = 8e3
y_max_ne = 8e5
y_min_mi = 9e-7
y_max_mi = 3e-3
y_min_ra = 5e-8 if pairs_only else 1e-13
y_max_ra = 2e-4 if pairs_only else 1e-3

# --- introduce model

# TODO: set axis limits to match anim

fig = plt.figure(figsize=(8, 4))
fig.supxlabel("Thousands of generations in past")

ne_ax = plt.subplot2grid((2, 2), (0, 0))
ne_ax.set_ylim(y_min_ne, y_max_ne)
ne_ax.set_xlim(x_min, x_max)
plot_ne(ne_ax, duration, params)
fig.tight_layout()

mi_ax = plt.subplot2grid((2, 2), (1, 0))
plot_migr(mi_ax, duration, params)
mi_ax.set_ylim(y_min_mi, y_max_mi)
mi_ax.set_xlim(x_min, x_max)
fig.tight_layout()

ra_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
plot_rates(ra_ax, duration, rates, pairs_only=True)
ra_ax.set_ylim(y_min_ra, y_max_ra)
ra_ax.set_xlim(x_min, x_max)
fig.tight_layout()

mi_ax.set_visible(False)
ra_ax.set_visible(False)
plt.savefig(output_dir + "pair-rates-0.png")

mi_ax.set_visible(True)
plt.savefig(output_dir + "pair-rates-1.png")

ra_ax.set_visible(True)
plt.savefig(output_dir + "pair-rates-2.png")

# --- describe model dynamics

highlight = [0, 20]
patches = []
patches.append(add_highlight(ne_ax, highlight))
patches.append(add_highlight(ra_ax, highlight))
patches.append(add_highlight(mi_ax, highlight))
plt.savefig(output_dir + "pair-rates-3.png")

for patch in patches: patch.remove()
highlight = [20, 40]
patches = []
patches.append(add_highlight(ne_ax, highlight))
patches.append(add_highlight(ra_ax, highlight))
patches.append(add_highlight(mi_ax, highlight))
plt.savefig(output_dir + "pair-rates-4.png")

for patch in patches: patch.remove()
highlight = [40, 50]
patches = []
patches.append(add_highlight(ne_ax, highlight))
patches.append(add_highlight(ra_ax, highlight))
patches.append(add_highlight(mi_ax, highlight))
plt.savefig(output_dir + "pair-rates-5.png")
