from core import *

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

# ------- #

output_dir = "../../fig/marg-ecdf/"
if not os.path.exists(output_dir): os.makedirs(output_dir)

ts = get_ts()
tree_subset = np.random.default_rng(1).choice(np.arange(ts.num_trees), 100)
tree_subset = [ts.at_index(t) for t in tree_subset]
root_length = 1e3
#x_lower = -3e2
#x_upper = np.max([t.time(t.root) for t in tree_subset]) + root_length
x_lower = 1e2
x_upper = 3e5

# ------ many ecdf ------ #

fig, axs = plt.subplots(figsize=(6, 3))

axs.set_xlim(x_lower, x_upper)
axs.set_ylim(-0.05, 1.01)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_ylabel("Empirical CDF")
axs.set_xlabel("Time in past")

# TODO: save this rather than recalculate it
tree_ecdf_subset = []
for tree in tree_subset:
    tree_ecdf, tree_times = marginal_ecdf(tree)
    assert tree_times[0] == 0 and tree_times[1] > 0
    tree_ecdf = np.append(tree_ecdf[1:], 1.0)
    tree_times = np.append(tree_times[1:], x_upper)
    tree_ecdf_subset.append((tree_times, tree_ecdf))

for tree, (tree_times, tree_ecdf) in zip(tree_subset, tree_ecdf_subset):
    axs.step(tree_times, tree_ecdf, where="post", alpha=0.3, linewidth=0.5, color="black")

axs.set_xscale('log') # maybe?

fig.tight_layout()
plt.savefig(output_dir + "marg-ecdf-1.png")

# I need to move to poppy to finish, with the expected CDF
