from core import *

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# ------- #

force_overwrite = False
output_dir = "../../fig/marg-ecdf/"
if not os.path.exists(output_dir): os.makedirs(output_dir)

ts = get_ts()
tree_subset = np.random.default_rng(1).choice(np.arange(ts.num_trees), 1000)
tree_subset = [ts.at_index(t) for t in tree_subset]

#root_length = 1e3
#x_lower = -3e2
#x_upper = np.max([t.time(t.root) for t in tree_subset]) + root_length

x_lower = 1e2 #log
x_upper = 3e5 #log

# ------ many ecdf ------ #

fig, axs = plt.subplots(figsize=(6, 3))

axs.set_xlim(x_lower, x_upper)
axs.set_ylim(-0.05, 1.01)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_ylabel(r"Empirical CDF")
axs.set_xlabel(r"Time in past $(t)$")

# TODO: save this rather than recalculate it
if os.path.exists("tmp.ecdf_marg.p") and not force_overwrite:
    marg_ecdf = pickle.load(open("tmp.ecdf_marg.p", "rb"))
    tree_ecdf = pickle.load(open("tmp.ecdf_tree.p", "rb"))
    true_ecdf = pickle.load(open("tmp.ecdf_true.p", "rb"))
else:
    marg_ecdf = average_ecdf(ts)
    pickle.dump(marg_ecdf, open("tmp.ecdf_marg.p", "wb"))
    tree_ecdf = []
    for tree in tree_subset:
        ecdf, times = marginal_ecdf(tree)
        assert times[0] == 0 and times[1] > 0
        ecdf = np.append(ecdf[1:], 1.0)
        times = np.append(times[1:], x_upper)
        tree_ecdf.append((times, ecdf))
    pickle.dump(tree_ecdf, open("tmp.ecdf_tree.p", "wb"))
    true_ecdf = theoretical_ecdf()
    pickle.dump(true_ecdf, open("tmp.ecdf_true.p", "wb"))

for times, ecdf in tree_ecdf:
    axs.step(times, ecdf, where="post", alpha=0.3, linewidth=0.5, color="black")

label = r'$\mathrm{ECDF}_{\mathcal{T}}(t)$ for trees $\mathcal{T}$ in ARG $\mathcal{S}$'
label = axs.text(0.01, 0.95, label, transform=axs.transAxes, ha='left', va='center')

axs.set_xscale('log') # maybe?

fig.tight_layout()
plt.savefig(output_dir + "marg-ecdf-1.png")

# ------ many ecdf pt 2 ------- #

for art in list(axs.lines): art.remove()
label.remove()

for times, ecdf in tree_ecdf:
    axs.step(times, ecdf, where="post", linewidth=0.5, color="gray", alpha=0.1)

axs.step(true_ecdf[0], true_ecdf[1], where="post", color="dodgerblue")
true_ecdf_label = r'Theoretical CDF'
true_ecdf_label = axs.text(0.01, 0.05, true_ecdf_label, transform=axs.transAxes, va='bottom', ha='left', color="dodgerblue")

label = r'$\mathrm{CDF}(t) = \int_{\mathcal{T} \in \mathbb{T}} \mathrm{ECDF}_{\mathcal{T}}(t) dP(\mathcal{T})$'
label = axs.text(0.01, 0.95, label, transform=axs.transAxes, ha='left', va='center')

plt.savefig(output_dir + "marg-ecdf-2.png")

# ------ many ecdf pt 3 ------- #

for art in list(axs.lines): art.remove()
label.remove()

axs.step(true_ecdf[0], true_ecdf[1], where="post", color="dodgerblue")
axs.step(marg_ecdf[0], marg_ecdf[1], where="post", color="black")
emp_ecdf_label = r'Marginalized ECDF'
emp_ecdf_label = axs.text(0.5, 0.45, emp_ecdf_label, transform=axs.transAxes, va='bottom', ha='right', color="black")

label = r'$\mathrm{CDF}(t) \approx \sum_{\mathcal{T} \in \mathcal{S}} \mathrm{ECDF}_{\mathcal{T}}(t) \frac{\mathrm{span}(\mathcal{T})}{ \mathrm{span}(\mathcal{S})}$'
label = axs.text(0.01, 0.95, label, transform=axs.transAxes, ha='left', va='center')

plt.savefig(output_dir + "marg-ecdf-3.png")

# ------ as a distribution ---- #

# ------ ecdf convergence ----- #

# ------ "node" view ---------- #
