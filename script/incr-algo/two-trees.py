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

# ------------------------------------ #

output_dir = "../../fig/incr-algo/"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# ---- two trees ---- #

fig, axs = plt.subplots(1, figsize=(6, 3))

axs.set_xlim(0, 9)
axs.set_ylim(0, 6)
axs.get_yaxis().set_visible(False)
axs.set_xticks([0, 4.5, 9], labels=[r"$0 \leftarrow$", r"Recombination event $(x)$", r"$\rightarrow L$"])
axs.spines['top'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.axvline(x=4.5, color='black', linestyle="--", linewidth=1)

draw_trio(axs, (1, 5), (0.5, 5.5), clade_width=0.5)
draw_trio(axs, (5, 9), (0.5, 5.5), clade_width=0.5, before_recomb=False)
plt.savefig(output_dir + "two-trees-0.png")

# ---- two trees with node labels ---- #

for art in list(axs.lines): art.remove()
for art in list(axs.patches): art.remove()

axs.axvline(x=4.5, color='black', linestyle="--", linewidth=1)

node_labels = []
coords = draw_trio(axs, (1, 5), (0.5, 5.5), clade_width=0.5)

node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$6$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$8$", color="black", ha="right", va="bottom"))

coords = draw_trio(axs, (5, 9), (0.5, 5.5), clade_width=0.5, before_recomb=False)
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [-0.1, 0.0]), r"$2$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$6$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$8$", color="black", ha="right", va="bottom"))

plt.savefig(output_dir + "two-trees-1.png")

# ---- two trees with node and edge labels ---- #

for art in list(axs.lines): art.remove()
for art in list(axs.patches): art.remove()

axs.axvline(x=4.5, color='black', linestyle="--", linewidth=1)

node_labels = []
coords = draw_trio(axs, (1, 5), (0.5, 5.5), clade_width=0.5, edge_highlight="firebrick")

node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$6$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$8$", color="black", ha="right", va="bottom"))

coords = draw_trio(axs, (5, 9), (0.5, 5.5), clade_width=0.5, before_recomb=False, edge_highlight="dodgerblue")
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [-0.1, 0.0]), r"$2$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$6$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$8$", color="black", ha="right", va="bottom"))

plt.savefig(output_dir + "two-trees-2.png")
