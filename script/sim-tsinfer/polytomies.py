from trio_core import *

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
axs.get_xaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.set_title(" ")

coords = draw_trio(axs, (3, 7), (0.5, 5.5), clade_width=0.5, edge_highlight="firebrick", highlight_path=False)

# --- node labels for samples ---
node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$4$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$5$", color="black", ha="right", va="bottom"))

plt.savefig(output_dir + "polytomies-0.png")

axs.text(0.02, 0.98, r"Sample counts $S_i$, pair counts $P_i$", ha='left', va='top', transform=axs.transAxes)
for lab in node_labels: lab.remove()
node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$P_4 = S_1 S_2$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$P_5 = S_4 S_3$", color="black", ha="right", va="bottom"))
plt.savefig(output_dir + "polytomies-1.png")

for lab in node_labels: lab.remove()
node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[3] + [-0.1, 0.0]), r"$P_4 = S_1 S_2$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$P_5 = S_1 S_3 + S_2 S_3$", color="black", ha="right", va="bottom"))
plt.savefig(output_dir + "polytomies-2.png")

for lab in node_labels: lab.remove()
for ln in list(axs.lines): ln.remove()
coords = draw_trio_poly(axs, (3, 7), (0.5, 5.5), clade_width=0.5, edge_highlight="firebrick", highlight_path=False)
node_labels = []
node_labels.append(axs.text(*(coords[0] + [-0.1, 0.0]), r"$1$", color="black", ha="right", va="bottom"))
node_labels.append(axs.text(*(coords[1] + [ 0.1, 0.0]), r"$2$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[2] + [ 0.1, 0.0]), r"$3$", color="black", ha="left", va="bottom"))
node_labels.append(axs.text(*(coords[5] + [-0.1, 0.0]), r"$P_5 = S_1 S_2 + S_1 S_3 + S_2 S_3$", color="black", ha="right", va="bottom"))
plt.savefig(output_dir + "polytomies-3.png")
