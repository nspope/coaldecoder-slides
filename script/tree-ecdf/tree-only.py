from core import *

import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage[svgnames]{xcolor}')


# ----------------------- #

output_dir = "../../fig/tree-ecdf/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
tree = get_tree()
root_length = 5e3
x_lower = -3.5e3
y_upper = len(list(tree.nodes()))
y_pad = 0


# ------ tree only ------ #

fig, axs = plt.subplots(1, figsize=(6, 3))

axs.set_ylim(-1, y_upper + y_pad)
axs.set_xlim(x_lower, tree.time(tree.root) + root_length)
axs.get_yaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.set_xlabel("Time in past")

draw_tree(axs, tree, root_length)

fig.tight_layout()
plt.savefig(output_dir + "tree-0.png")


# ------ tree with lines ------ #

for art in list(axs.lines): art.remove()

tmrca = [tree.time(i) for i in tree.nodes() if not tree.is_leaf(i)]
label = r"Ancestors $\implies$ coalescence events"
for t in tmrca:
    axs.axvline(x=t, ymin=0, ymax=1.0, linewidth=0.75, color='firebrick', alpha=0.5) #linestyle='--')

draw_tree(axs, tree, root_length)

axs.text(0.35, 0.9, label, ha='left', va='bottom', transform=axs.transAxes, size=12, color='firebrick')
plt.savefig(output_dir + "tree-1.png")
