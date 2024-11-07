
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
tree = ts.first()
root_length = 1e3
x_lower = -3e2
x_upper = tree.time(tree.root) + root_length


# ------ one tree and ecdf ------ #

fig = plt.figure(figsize=(6, 3.5))
axs_tree = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
axs_ecdf = plt.subplot2grid((3, 1), (1, 0), rowspan=2)

axs_tree.set_ylim(-1, len(list(tree.nodes())))
axs_tree.set_xlim(x_lower, x_upper)
axs_tree.get_yaxis().set_visible(False)
axs_tree.get_xaxis().set_visible(False)
axs_tree.spines['top'].set_visible(False)
axs_tree.spines['right'].set_visible(False)
axs_tree.spines['left'].set_visible(False)
axs_tree.spines['bottom'].set_visible(False)
draw_tree(axs_tree, ts.first(), root_length, x_lower)

axs_ecdf.set_xlim(x_lower, x_upper)
axs_ecdf.set_ylim(-0.05, 1.01)
axs_ecdf.spines['top'].set_visible(False)
axs_ecdf.spines['right'].set_visible(False)
axs_ecdf.set_ylabel("Empirical CDF")
axs_ecdf.set_xlabel("Generations in past")
tree_ecdf, tree_times = marginal_ecdf(tree)
tree_ecdf = np.append(tree_ecdf, tree_ecdf[-1])
tree_times = np.append(tree_times, tree_times[-1] + root_length)
axs_ecdf.step(tree_times, tree_ecdf, where="post", color='black')
#draw_pair(axs_ecdf, (0.05, 0.10), (0.87, 0.95), labels=["CEU", "CHB"], color='black', label_color=["firebrick", "dodgerblue"])
axs_ecdf.text(0.01, 0.95, r"(CEU, CHB) pairs", fontsize=8)

tubes_inset = draw_tubes(axs_ecdf, [.4, .15, .2, .4])

fig.tight_layout()
plt.savefig(output_dir + "marg-ecdf-0.png")


