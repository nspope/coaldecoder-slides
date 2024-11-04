import tskit
import numpy as np
import msprime
import itertools


def get_tree() -> tskit.Tree:
    seed = 1
    reps = 1
    ne = 2e4
    t1 = 5e3
    t2 = 7e3
    t3 = 15e3
    m0 = 1e-3
    m1 = 5e-3
    n0 = 10
    n1 = 10
    pop_config = [
        msprime.PopulationConfiguration(sample_size=n0), 
        msprime.PopulationConfiguration(sample_size=n1),
    ]
    demography = [
        msprime.MigrationRateChange(time=0.0, rate=m0), 
        msprime.MigrationRateChange(time=t1, matrix_index=(0, 1), rate=m1),
        msprime.MigrationRateChange(time=t2, matrix_index=(0, 1), rate=m0),
        msprime.MassMigration(time=t3, source=1, dest=0)
    ]
    ts = msprime.simulate(
        population_configurations=pop_config,
        demographic_events=demography,
        Ne=ne / 2.,
        length=1,
        recombination_rate=0,
        random_seed=seed,
    )
    return ts.first()


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

    labelling = {i : ('A' if traversal_position[i] < (height - 1) / 2 else 'B') for i in tree.samples() }

    return traversal_position, height, labelling


def count_pairs(tree: tskit.Tree) -> dict:
    pairs = {}
    for n in tree.nodes():
        pairs[n] = 0
        for u, v in itertools.combinations(tree.children(n), 2):
            pairs[n] += tree.num_samples(u) * tree.num_samples(v)
    return pairs


def count_pairs_across(tree: tskit.Tree) -> dict:
    *_, labelling = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([labelling[i] == 'A' for i in tree.samples(n)])
        S_B[n] = np.sum([labelling[i] == 'B' for i in tree.samples(n)])
    P_AB = {}
    for n in tree.nodes():
        P_AB[n] = 0
        for u, v in itertools.combinations(tree.children(n), 2):
            P_AB[n] += S_A[u] * S_B[v] + S_A[v] * S_B[u]
    return P_AB


def count_trios_ABA(tree: tskit.Tree) -> dict:
    *_, labelling = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([labelling[i] == 'A' for i in tree.samples(n)])
        S_B[n] = np.sum([labelling[i] == 'B' for i in tree.samples(n)])
    T_ABA = {}
    for n in tree.nodes():
        T_ABA[n] = 0.0
        for u, v in itertools.combinations(tree.children(n), 2):
            T_ABA[n] += S_A[v] * S_A[u] * S_B[u] + S_A[u] * S_A[v] * S_B[v]
    return T_ABA


def count_trios_AAB(tree: tskit.Tree) -> dict:
    *_, labelling = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([labelling[i] == 'A' for i in tree.samples(n)])
        S_B[n] = np.sum([labelling[i] == 'B' for i in tree.samples(n)])
    T_AAB = {}
    for n in tree.nodes():
        T_AAB[n] = 0.0
        for u, v in itertools.combinations(tree.children(n), 2):
            T_AAB[n] += S_B[v] * S_A[u] * (S_A[u] - 1) / 2 + S_B[u] * S_A[v] * (S_A[v] - 1) / 2
    return T_AAB

def count_trios_ABB(tree: tskit.Tree) -> dict:
    *_, labelling = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([labelling[i] == 'A' for i in tree.samples(n)])
        S_B[n] = np.sum([labelling[i] == 'B' for i in tree.samples(n)])
    T_ABB = {}
    for n in tree.nodes():
        T_ABB[n] = 0.0
        for u, v in itertools.combinations(tree.children(n), 2):
            T_ABB[n] += S_B[v] * S_A[u] * S_B[u] + S_B[u] * S_A[v] * S_B[v]
    return T_ABB

def count_trios_BBA(tree: tskit.Tree) -> dict:
    *_, labelling = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([labelling[i] == 'A' for i in tree.samples(n)])
        S_B[n] = np.sum([labelling[i] == 'B' for i in tree.samples(n)])
    T_BBA = {}
    for n in tree.nodes():
        T_BBA[n] = 0.0
        for u, v in itertools.combinations(tree.children(n), 2):
            T_BBA[n] += S_A[v] * S_B[u] * (S_B[u] - 1) / 2 + S_A[u] * S_B[v] * (S_B[v] - 1) / 2
    return T_BBA


def count_trios_AAB(tree: tskit.Tree) -> dict:
    *_, labelling = assign_traversal_positions(tree)
    S_A = {}
    S_B = {}
    for n in tree.nodes():
        S_A[n] = np.sum([labelling[i] == 'A' for i in tree.samples(n)])
        S_B[n] = np.sum([labelling[i] == 'B' for i in tree.samples(n)])
    T_AAB = {}
    for n in tree.nodes():
        T_AAB[n] = 0.0
        for u, v in itertools.combinations(tree.children(n), 2):
            T_AAB[n] += S_B[v] * S_A[u] * (S_A[u] - 1) / 2 + S_B[u] * S_A[v] * (S_A[v] - 1) / 2
    return T_AAB


def draw_tree(axs, tree, root_length):
    node_y, height, labelling = assign_traversal_positions(tree)
    line_kwargs = { 'c' : 'black', 'linewidth' : 1 }
    for n in tree.postorder():
        p = tree.parent(n)
        if p == tskit.NULL: 
            axs.plot((tree.time(n), tree.time(n) + root_length), (node_y[n], node_y[n]), **line_kwargs)
        else:
            axs.plot((tree.time(n), tree.time(p)), (node_y[n], node_y[n]), **line_kwargs)
            axs.plot((tree.time(p), tree.time(p)), (node_y[n], node_y[p]), **line_kwargs)


def draw_clade_labels(axs, bar_offset, bar_gap):
    line_kwargs = { 'c' : 'black', 'linewidth' : 2 }
    text_kwargs = { 'size' : 12, 'ha' : 'left' }
    axs.plot((bar_offset, bar_offset), (0.0, 0.5 - bar_gap), transform=axs.transAxes, **line_kwargs)
    axs.plot((bar_offset, bar_offset), (0.5 + bar_gap, 1.0), transform=axs.transAxes, **line_kwargs)
    axs.text(0., 0.2, "A", transform=axs.transAxes, **text_kwargs)
    axs.text(0., 0.7, "B", transform=axs.transAxes, **text_kwargs)


def draw_ecdf(axs, tree, node_weights, root_length, color='black', denom=None):
    times = []
    weights = []
    for u in tree.nodes():
        times.append(tree.time(u))
        weights.append(node_weights[u])
    time_points = np.unique(times)
    time_index = np.digitize(times, time_points) - 1
    weights = np.bincount(time_index, weights=weights)
    if denom is None: denom = np.sum(weights)
    ecdf = np.cumsum(weights) / denom
    time_points = np.append(time_points, time_points[-1] + root_length)
    ecdf = np.append(ecdf, ecdf[-1])
    return axs.step(time_points, ecdf, color=color, where='post')


def draw_node_weights(axs, tree, node_weights, denom=None, color='black', scale=25):
    node_y, height, labelling = assign_traversal_positions(tree)
    if denom is None: 
        #denom = np.max([w for w in node_weights.values()])
        denom = np.sum([w for w in node_weights.values()])
    node_size = {n: w / denom for n, w in node_weights.items()}
    points = []
    for n in tree.postorder():
        points += axs.plot(tree.time(n), node_y[n], color=color, marker='o', markersize=node_size[n] * scale)
    return points


def draw_pair(axs, x, y, labels=None, color='black'):
    xmin, xmax = x
    ymin, ymax = y
    root = (xmin + xmax) / 2
    axs.plot((xmax, xmax), (ymin, ymax), color=color, transform=axs.transAxes)
    axs.plot((xmax, root), (ymin, ymin), color=color, transform=axs.transAxes)
    axs.plot((xmin, xmin), (ymin, ymax), color=color, transform=axs.transAxes)
    axs.plot((xmin, root), (ymin, ymin), color=color, transform=axs.transAxes)
    axs.plot((root, root), (ymin, ymin - (ymax - ymin) * 0.75), color=color, transform=axs.transAxes)
    axs.plot(root, ymin, marker="o", markersize=4, color=color, transform=axs.transAxes)
    tip_labels = []
    if labels is not None:
        assert len(labels) == 2
        tip_labels += [axs.text(xmin, ymax, labels[0], va='bottom', ha='center', color=color, transform=axs.transAxes)]
        tip_labels += [axs.text(xmax, ymax, labels[1], va='bottom', ha='center', color=color, transform=axs.transAxes)]
    return tip_labels


def draw_trio(axs, x, y, labels=None, left_first=True, color='black'):
    xmin, xmax = x
    ymin, ymax = y
    y0, y1, y2 = ymin, (ymax + ymin) / 2, ymax
    x0, x1, x2 = xmin, (xmax + xmin) / 2, xmax
    if not left_first:
        x0, x1, x2 = x2, x1, x0
    r1 = (x0 + x1) / 2
    r2 = (r1 + x2) / 2
    axs.plot((x2, x2), (y0, y2), color=color, transform=axs.transAxes)
    axs.plot((x1, x1), (y1, y2), color=color, transform=axs.transAxes)
    axs.plot((x0, x0), (y1, y2), color=color, transform=axs.transAxes)
    axs.plot((x0, x1), (y1, y1), color=color, transform=axs.transAxes)
    axs.plot((r1, r1), (y1, y0), color=color, transform=axs.transAxes)
    axs.plot((r1, x2), (y0, y0), color=color, transform=axs.transAxes)
    axs.plot((r2, r2), (y0, y0 - (y2 - y0) * 0.75), color=color, transform=axs.transAxes)
    axs.plot(r2, ymin, marker="o", markersize=4, color=color, transform=axs.transAxes)
    tip_labels = []
    if labels is not None:
        assert len(labels) == 3
        tip_labels += [axs.text(xmin, y2, labels[0], va='bottom', ha='center', color=color, transform=axs.transAxes)]
        tip_labels += [axs.text((xmin + xmax) / 2, y2, labels[1], va='bottom', ha='center', color=color, transform=axs.transAxes)]
        tip_labels += [axs.text(xmax, y2, labels[2], va='bottom', ha='center', color=color, transform=axs.transAxes)]
    return tip_labels
