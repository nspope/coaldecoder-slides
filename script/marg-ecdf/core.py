import tskit
import os
import numpy as np
import stdpopsim
import msprime
import itertools


def get_ts() -> tskit.TreeSequence:
    if os.path.exists("tmp.trees"):
        return tskit.load("tmp.trees")
    else:
        homsap = stdpopsim.get_species("HomSap")
        demogr = homsap.get_demographic_model("OutOfAfrica_3G09")
        sample = {"YRI" : 125, "CEU" : 125, "CHB" : 125}
        contig = homsap.get_contig("chr2", left=180e6, right=230e6)
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(contig=contig, demographic_model=demogr, samples=sample, random_seed=1)
        ts.dump("tmp.trees")
        return ts


def assign_traversal_positions(tree: tskit.Tree) -> dict:
    traversal_position = {}
    traversal_order = "minlex_postorder"

    # taking from tskit.drawing
    y = 0
    for u in tree.nodes(tree.root, order=traversal_order):
        if tree.is_leaf(u):
            traversal_position[u] = y
            y += 2
        else:
            coords = [traversal_position[c] for c in tree.children(u)]
            if len(coords) == 1:
                traversal_position[u] = coords[0]
            else:
                a = min(coords)
                b = max(coords)
                child_mid = int(round(a + (b - a) / 2))
                traversal_position[u] = child_mid
    y += 1
    height = y - 2

    # reorder so larger clade is above smaller
    for n in tree.postorder():
        if tree.num_children(n) == 0: continue
        u, v = list(tree.children(n))
        bigger = list(tree.nodes(u))
        smaller = list(tree.nodes(v))
        if len(bigger) == len(smaller): continue
        if len(bigger) < len(smaller):
            bigger, smaller = smaller, bigger
        if not min([traversal_position[i] for i in bigger]) > max([traversal_position[i] for i in smaller]):
            bigger_lo = min([traversal_position[i] for i in bigger])
            bigger_hi = max([traversal_position[i] for i in bigger])
            smaller_lo = min([traversal_position[i] for i in smaller])
            smaller_hi = max([traversal_position[i] for i in smaller])
            for i in smaller:
                traversal_position[i] += -smaller_lo + bigger_lo
            for i in bigger:
                traversal_position[i] += -bigger_hi + smaller_hi
            assert min([traversal_position[i] for i in bigger]) > max([traversal_position[i] for i in smaller])
        traversal_position[n] = traversal_position[u] / 2 + traversal_position[v] / 2

    ts = tree.tree_sequence
    pop_names = { i : ts.population(i).metadata['name'] for i in range(ts.num_populations) }
    label = { i : pop_names[ts.nodes_population[i]] for i in tree.samples() }

    return traversal_position, height, label


def count_pairs_across(tree: tskit.Tree) -> dict:
    *_, label = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([label[i] == 'CEU' for i in tree.samples(n)])
        S_B[n] = np.sum([label[i] == 'CHB' for i in tree.samples(n)])
    P_AB = {}
    for n in tree.nodes():
        P_AB[n] = 0
        for u, v in itertools.combinations(tree.children(n), 2):
            P_AB[n] += S_A[u] * S_B[v] + S_A[v] * S_B[u]
    return P_AB


def theoretical_ecdf():
    breakpoints = np.logspace(0, 6, 10000)
    breakpoints[0] = 0
    homsap = stdpopsim.get_species("HomSap")
    demogr = homsap.get_demographic_model("OutOfAfrica_3G09")
    debug = demogr.model.debug()
    _, pair_cdf = debug.coalescence_rate_trajectory(lineages={"CEU":1, "CHB":1}, steps=breakpoints)
    pair_cdf = 1 - pair_cdf
    return breakpoints, pair_cdf


def marginal_ecdf(tree):
    counts = count_pairs_across(tree)
    weights = np.array([counts[i] for i in tree.nodes()], dtype=float)
    weights /= np.sum(weights)
    times = np.array([tree.time(i) for i in tree.nodes()])
    node_times = np.unique(times)
    node_index = np.digitize(times, node_times) - 1
    assert np.all(np.logical_and(node_index >= 0, node_index < node_times.size))
    node_weights = np.bincount(node_index, weights=weights, minlength=node_times.size)
    node_weights = np.cumsum(node_weights)
    assert np.isclose(node_weights[-1], 1.0)
    return node_weights, node_times


def average_ecdf(ts):
    pop_index = { ts.population(i).metadata['name'] : i for i in range(ts.num_populations) }
    sample_sets = [
        np.flatnonzero(ts.nodes_population[:ts.num_samples] == pop_index["CEU"]),
        np.flatnonzero(ts.nodes_population[:ts.num_samples] == pop_index["CHB"]),
    ]
    node_weights = ts.pair_coalescence_counts(sample_sets, pair_normalise=True)
    node_times = np.unique(ts.nodes_time)
    node_index = np.digitize(ts.nodes_time, node_times) - 1
    assert np.all(np.logical_and(node_index >= 0, node_index < node_times.size))
    node_weights = np.bincount(node_index, weights=node_weights, minlength=node_times.size)
    node_weights = np.cumsum(node_weights)
    assert np.isclose(node_weights[-1], 1.0)
    return node_times, node_weights 


def draw_tree(axs, tree, root_length, point_x):
    node_y, height, labeling = assign_traversal_positions(tree)
    line_kwargs = { 'c' : 'black', 'linewidth' : 0.5 }
    for n in tree.postorder():
        p = tree.parent(n)
        if p == tskit.NULL: 
            axs.plot((tree.time(n), tree.time(n) + root_length), (node_y[n], node_y[n]), **line_kwargs)
        else:
            axs.plot((tree.time(n), tree.time(p)), (node_y[n], node_y[n]), **line_kwargs)
            axs.plot((tree.time(p), tree.time(p)), (node_y[n], node_y[p]), **line_kwargs)
        if tree.is_leaf(n):
            if labeling[n] == "CEU":
                axs.plot(point_x, node_y[n], marker="s", markersize=1, color="firebrick", label="CEU")
            elif labeling[n] == "CHB":
                axs.plot(point_x, node_y[n], marker="s", markersize=1, color="dodgerblue", label="CHB")

def draw_pair(axs, x, y, labels=None, label_color=None, color='black'):
    xmin, xmax = x
    ymin, ymax = y
    root = (xmin + xmax) / 2
    line_kwargs = {"color" : color, "linewidth" : 1}
    axs.plot((xmax, xmax), (ymin, ymax), transform=axs.transAxes, **line_kwargs)
    axs.plot((xmax, root), (ymin, ymin), transform=axs.transAxes, **line_kwargs)
    axs.plot((xmin, xmin), (ymin, ymax), transform=axs.transAxes, **line_kwargs)
    axs.plot((xmin, root), (ymin, ymin), transform=axs.transAxes, **line_kwargs)
    axs.plot((root, root), (ymin, ymin - (ymax - ymin) * 0.75), transform=axs.transAxes, **line_kwargs)
    axs.plot(root, ymin, marker="o", markersize=4, color=color, transform=axs.transAxes)
    tip_labels = []
    if labels is not None:
        assert len(labels) == 2
        assert len(label_color) == 2
        tip_labels += [axs.text(xmin, ymax + 0.01, labels[0], va='bottom', ha='center', color=label_color[0], size=6, transform=axs.transAxes)]
        tip_labels += [axs.text(xmax, ymax + 0.01, labels[1], va='bottom', ha='center', color=label_color[1], size=6, transform=axs.transAxes)]
    return tip_labels


def draw_tubes(axs, pos):
    import demesdraw
    homsap = stdpopsim.get_species("HomSap")
    demogr = homsap.get_demographic_model("OutOfAfrica_3G09").model.to_demes()
    sub_axs = axs.inset_axes(pos)
    demesdraw.tubes(
        demogr, 
        sub_axs, 
        num_lines_per_migration=0, 
        fill=True,
        colours={"YRI":"black", "CEU":"firebrick", "CHB":"dodgerblue"},
    )
    sub_axs.get_yaxis().set_visible(False)
    sub_axs.spines['left'].set_visible(False)
    sub_axs.set_xticklabels(labels=[r"\ttfamily{YRI}", r"\ttfamily{CEU}", r"\ttfamily{CHB}"], fontsize=8)
    sub_axs.set_title(r"\ttfamily{OutOfAfrica_3G09}", size=8)
    return sub_axs
