
from core import *

import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')


# ----------------------- #

output_dir = "../../fig/tree-ecdf/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
tree = get_tree()
root_length = 5e3
x_lower = -3.5e3
x_upper = tree.time(tree.root) + root_length


# ------ standard weighting ------ #

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
draw_tree(axs_tree, tree, root_length)

#tmrca = [tree.time(i) for i in tree.nodes() if not tree.is_leaf(i)]
#for t in tmrca:
#    axs_tree.axvline(x=t, ymin=0, ymax=1.0, linewidth=0.75, color='firebrick', alpha=0.5)

axs_ecdf.set_xlim(x_lower, x_upper)
axs_ecdf.set_ylim(-0.05, 1.01)
axs_ecdf.spines['top'].set_visible(False)
axs_ecdf.spines['right'].set_visible(False)
axs_ecdf.set_ylabel("Empirical CDF")
axs_ecdf.set_xlabel("Time in past")
uniform_weights = {i : (0 if tree.is_leaf(i) else 1) for i in tree.nodes() }
draw_ecdf(axs_ecdf, tree, uniform_weights, root_length, color='black')

fig.tight_layout()
plt.savefig(output_dir + "tree-ecdf-0.png")


# ------ pair weights ------ #

for art in list(axs_ecdf.lines): art.remove()

draw_ecdf(axs_ecdf, tree, count_pairs(tree), root_length, color='black')
draw_node_weights(axs_tree, tree, count_pairs(tree), color='black')
label = r"$\mathrm{Pairs(}i) = 2^{-1} \sum_{j \in \mathcal{C}(i)} " + \
    r"\mathrm{Samples(}j) \times (\mathrm{Samples(}i) - \mathrm{Samples(}j))$"
label = axs_ecdf.text(0.0, 0.99, label, ha='left', va='top', size=12, usetex=True)

plt.savefig(output_dir + "tree-ecdf-1.png")


# ------ clade labels ------ #

for art in list(axs_tree.lines): art.remove()
#for art in list(axs_ecdf.lines): art.remove()
#label.remove()

draw_clade_labels(axs_tree, 0.03, 0.03)
draw_tree(axs_tree, tree, root_length)

plt.savefig(output_dir + "tree-ecdf-2.png")


# ------ cross-coalescence ------ #

for art in list(axs_ecdf.lines): art.remove()
label.remove()

draw_ecdf(axs_ecdf, tree, count_pairs_across(tree), root_length, color='black')
draw_node_weights(axs_tree, tree, count_pairs_across(tree), color='black')

label = r"$P^{AB}_i = 2^{-1} \sum_{j \in \mathcal{C}(i)} " + \
    r"S^{A}_j (S^{B}_i - S^{B}_j) + S^{B}_j (S^{A}_i - S^{A}_j) $"
label = axs_ecdf.text(0.0, 0.99, label, ha='left', va='top', size=12, usetex=True)

tip_labels = draw_pair(axs_ecdf, (0.85, 0.9), (0.38, 0.48), ["A", "B"])

plt.savefig(output_dir + "tree-ecdf-3.png")


# ------ trio cross-coalescence pt 1 ------ #

for art in list(axs_tree.lines): art.remove()
for art in list(axs_ecdf.lines): art.remove()
for text in tip_labels: text.remove()
label.remove()

color = 'firebrick'
ABB_counts = count_trios_ABB(tree)
BBA_counts = count_trios_BBA(tree)
ABB_BBA_totals = np.sum([x for x in ABB_counts.values()]) + \
    np.sum([x for x in BBA_counts.values()])
assert ABB_BBA_totals == 450

draw_clade_labels(axs_tree, 0.03, 0.03)
draw_tree(axs_tree, tree, root_length)
draw_node_weights(axs_tree, tree, ABB_counts, denom=ABB_BBA_totals, color=color)
draw_ecdf(axs_ecdf, tree, ABB_counts, root_length, denom=ABB_BBA_totals, color=color)
label = r"$T^{AB,B}_i = 2^{-1} \sum_{j \in \mathcal{C}(i)} " + \
    r"S^{B}_j (S^{A}_i - S^{A}_j) (S^{B}_i - S^{B}_j) + (S^{B}_i - S^{B}_j) S^{A}_j S^{B}_j$"
label = axs_ecdf.text(0.0, 0.99, label, ha='left', va='top', size=12, usetex=True, color='black')
draw_trio(axs_ecdf, (0.93, 0.98), (0.25, 0.35), ["A", "B", "B"], left_first=True, color=color)

plt.savefig(output_dir + "tree-ecdf-4.png")

# ------ trio cross-coalescence pt 2 ------ #

for art in list(axs_tree.lines): art.remove()
label.remove()

color = 'dodgerblue'

draw_clade_labels(axs_tree, 0.03, 0.03)
draw_tree(axs_tree, tree, root_length)
draw_node_weights(axs_tree, tree, BBA_counts, denom=ABB_BBA_totals, color=color)
draw_ecdf(axs_ecdf, tree, BBA_counts, root_length, denom=ABB_BBA_totals, color=color)
label = r"$T^{BB,A}_i = 2^{-1} \sum_{j \in \mathcal{C}(i)} " + \
    r"S^{A}_j {S^{B}_i - S^{B}_j \choose 2} + (S^{A}_i - S^{A}_j) {S^{B}_j \choose 2}$"
label = axs_ecdf.text(0.0, 0.99, label, ha='left', va='top', size=12, usetex=True, color='black')
draw_trio(axs_ecdf, (0.87, 0.92), (0.75, 0.85), ["A", "B", "B"], left_first=False, color=color)

plt.savefig(output_dir + "tree-ecdf-5.png")
