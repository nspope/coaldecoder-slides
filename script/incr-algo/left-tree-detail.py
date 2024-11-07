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

# ---- one tree with node labels ---- #

fig, axs = plt.subplots(1, figsize=(6, 3))

axs.set_xlim(0, 9)
axs.set_ylim(0, 6)
axs.get_yaxis().set_visible(False)
axs.set_xticks([0, 6.5, 9], labels=[r"$0 \leftarrow$", r"Recombination event $(x)$", r"$\rightarrow L$"])
axs.spines['top'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)

coords = draw_trio(axs, (3, 7), (0.5, 5.5), clade_width=0.5, edge_highlight="firebrick", highlight_path=False)
axs.axvline(x=6.5, color='black', linestyle="--", linewidth=1)

# --- node labels for samples ---
node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$6$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$8$", color="black", ha="right", va="bottom"))

plt.savefig(output_dir + "left-tree-detail-0.png")

# --- node labels for samples ---
for lab in node_labels: lab.remove()
node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$S_1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$S_2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$S_3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$S_4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$S_5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$S_6$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$S_7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$S_8$", color="black", ha="right", va="bottom"))

plt.savefig(output_dir + "left-tree-detail-1.png")

# --- node labels for updated samples ---
for lab in node_labels: lab.remove()
for art in list(axs.lines): art.remove()
for patch in list(axs.patches): patch.remove()

coords = draw_trio(axs, (3, 7), (0.5, 5.5), clade_width=0.5, edge_highlight="firebrick", highlight_path=True)
axs.axvline(x=6.5, color='black', linestyle="--", linewidth=1)

node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$S_1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$S_2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$S_3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$S_4' = S_4 - S_2$", color="firebrick", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$S_5$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$S_6' = S_6 - S_2$", color="firebrick", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$S_7$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$S_8' = S_8 - S_2$", color="firebrick", ha="right", va="bottom"))

plt.savefig(output_dir + "left-tree-detail-2.png")


# --- node labels for pairs ---
for lab in node_labels: lab.remove()

node_labels = []
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$P_4 = S_2 (S_4 - S_2)$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$P_6 = S_4 (S_6 - S_4)$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$P_8 = S_6 (S_8 - S_6)$", color="black", ha="right", va="bottom"))
#node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
#node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
#node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$5$", color="black", ha="left", va="bottom"))
#node_labels.append(axs.text(*(coords[6] + [ 0.1, 0.0]), r"$7$", color="black", ha="left", va="bottom"))

plt.savefig(output_dir + "left-tree-detail-3.png")

# --- node labels for updated pairs ---
for lab in node_labels: lab.remove()

node_labels = []
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$P_4' = P_4 - S_2 (S_4 - S_2)$", color="firebrick", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$P_6' = P_6 - S_2 (S_6 - S_4)$", color="firebrick", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$P_8' = P_8 - S_2 (S_8 - S_6)$", color="firebrick", ha="right", va="bottom"))

plt.savefig(output_dir + "left-tree-detail-4.png")

# --- node labels for updated expectation ---
for lab in node_labels: lab.remove()

node_labels = []
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$\mathbb{E}[P_4] \leftarrow \mathbb{E}[P_4] - \omega S_2 (S_4 - S_2)$", color="firebrick", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$\mathbb{E}[P_6] \leftarrow \mathbb{E}[P_6] - \omega S_2 (S_6 - S_4)$", color="firebrick", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$\mathbb{E}[P_8] \leftarrow \mathbb{E}[P_8] - \omega S_2 (S_8 - S_6)$", color="firebrick", ha="right", va="bottom"))
annot = axs.text(0.15, 0.2, r"$\omega = \frac{L - x}{L}$", transform=axs.transAxes, va='center', ha='center')

plt.savefig(output_dir + "left-tree-detail-5.png")
