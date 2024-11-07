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
axs.set_xticks([0, 2.5, 9], labels=[r"$0 \leftarrow$", r"Recombination event $(x)$", r"$\rightarrow L$"])
axs.spines['top'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)

coords = draw_trio(axs, (3, 7), (0.5, 5.5), clade_width=0.5, edge_highlight="dodgerblue", before_recomb=False, highlight_path=True, fade_outgroup=True)
axs.axvline(x=2.5, color='black', linestyle="--", linewidth=1, alpha=0.1)

# --- updates only ---- #

node_labels = []
node_labels.append(axs.text(*(coords[1] + [-0.1, 0.0]), r"$2$", color="black", ha="right", va="bottom"))
#node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$\mathbb{E}[P_5] \leftarrow \mathbb{E}[P_5] + \omega S_2 (S_5 - S_2)$", color="dodgerblue", ha="right", va="bottom"))
#node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$\mathbb{E}[P_6] \leftarrow \mathbb{E}[P_6] + \omega S_2 (S_6 - S_5)$", color="dodgerblue", ha="right", va="bottom"))
#node_labels.append(axs.text(*(coords[7] + [-0.1, 0.0]), r"$\mathbb{E}[P_8] \leftarrow \mathbb{E}[P_8] + \omega S_2 (S_8 - S_6)$", color="dodgerblue", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[4] + [ 0.1, 0.0]), r"$\mathbb{E}[P_5] \leftarrow \mathbb{E}[P_5] + \omega S_2 (S_5 - S_2)$", color="dodgerblue", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$\mathbb{E}[P_6] \leftarrow \mathbb{E}[P_6] + \omega S_2 (S_6 - S_5)$", color="dodgerblue", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[7] + [ 0.1, 0.0]), r"$\mathbb{E}[P_8] \leftarrow \mathbb{E}[P_8] + \omega S_2 (S_8 - S_6)$", color="dodgerblue", ha="left", va="bottom"))
annot = axs.text(0.15, 0.2, r"$\omega = \frac{L - x}{L}$", transform=axs.transAxes, va='center', ha='center')

plt.savefig(output_dir + "right-tree-detail-0.png")

