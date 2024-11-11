import tskit
import numpy as np
import msprime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

out_dir = "../../fig/arg-3d/"


# ---

def nodes_above(ts, node):
    traverse_up = np.full(ts.num_nodes, False)
    traverse_up[node] = True
    for p, c in zip(ts.edges_parent, ts.edges_child):
        traverse_up[p] |= traverse_up[c]
    return traverse_up

def nodes_below(ts, node):
    traverse_down = np.full(ts.num_nodes, False)
    traverse_down[node] = True
    for p, c in zip(ts.edges_parent[::-1], ts.edges_child[::-1]):
        traverse_down[c] |= traverse_down[p]
    return traverse_down


focal_node = 200
ts = msprime.sim_ancestry(samples=50, sequence_length=1e6, recombination_rate=1e-8, population_size=1e4, random_seed=1)
above = nodes_above(ts, focal_node)
below = nodes_below(ts, focal_node)

fig, axs = plt.subplots(1, figsize=(6, 3))
axs.set_xlabel("Position on sequence (Mb)")
axs.set_ylabel("Node age")

for c, l, r in zip(ts.edges_child, ts.edges_left/1e6, ts.edges_right/1e6):
    t = ts.nodes_time[c]
    if t > 0:
        axs.plot((l, r), (t, t), color="gray", alpha=0.5, linewidth=0.5)
axs.set_xlim(0, 1)
axs.set_yscale('log')
title = axs.set_title("``Node'' view")
fig.tight_layout()
plt.savefig(out_dir + "node-view-0.png")

labelled_above = False
labelled_below = False
highlight = []
for c, l, r in zip(ts.edges_child, ts.edges_left/1e6, ts.edges_right/1e6):
    t = ts.nodes_time[c]
    if c == focal_node:
        focal, *_ = axs.plot((l, r), (t, t), color="firebrick", linewidth=1, label="Focal node")
    elif above[c]:
        ln, *_ = axs.plot((l, r), (t, t), color="dodgerblue", linewidth=0.5, label="On paths to roots" if not labelled_above else None)
        highlight.append(ln)
        labelled_above = True
    elif below[c]:
        ln, *_ = axs.plot((l, r), (t, t), color="forestgreen", linewidth=0.5, label="On paths to leaves" if not labelled_below else None)
        highlight.append(ln)
        labelled_below = True
for art in highlight: art.set_visible(False)
plt.savefig(out_dir + "node-view-1.png")

for art in highlight: art.set_visible(True)
axs.legend(framealpha=1.0, loc='lower left')
title.set_text("Complex dependence structure!")
plt.savefig(out_dir + "node-view-2.png")
