import tskit
import pickle
import os
import numpy as np
import msprime
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

def get_scaling(ts):
    edges_visited = 0
    naive_edges = np.sum([t.num_edges for t in ts.trees()])
    nodes_parent = np.full(ts.num_nodes, tskit.NULL)
    for ed in ts.edge_diffs():
        for e in ed.edges_out:
            p = e.parent
            c = e.child
            nodes_parent[c] = tskit.NULL
            while p != tskit.NULL:
                edges_visited += 1
                p = nodes_parent[p]

        for e in ed.edges_in:
            p = e.parent
            c = e.child
            nodes_parent[c] = p
            while p != tskit.NULL:
                edges_visited += 1
                p = nodes_parent[p]

    return edges_visited / ts.num_trees, naive_edges / ts.num_trees

# generate data
data_store = "tmp.scaling.p"
if not os.path.exists(data_store):
    sample_grid = np.logspace(1, 5, 20)
    n_samples = []
    n_edges_fast = []
    n_edges_naive = []
    for s in sample_grid:
        ts = msprime.sim_ancestry(int(s), sequence_length=1e7, recombination_rate=1e-8, population_size=1e4, random_seed=int(s))
        fast, naive = get_scaling(ts)
        n_edges_fast.append(fast)
        n_edges_naive.append(naive)
        n_samples.append(s)
    pickle.dump((n_samples, n_edges_fast, n_edges_naive), open(data_store, "wb"))
else:
    (n_samples, n_edges_fast, n_edges_naive) = pickle.load(open(data_store, "rb"))

import matplotlib.pyplot as plt
out_dir = "../../fig/incr-algo/"
plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
plt.plot(n_samples, n_edges_fast, "-o", color="dodgerblue", label="Incremental")
plt.plot(n_samples, n_edges_naive, "-o", color="firebrick", label="Naive")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of diploids")
plt.ylabel("Mean edges visited per tree")
plt.legend()
plt.title("Scaling across numbers of tips", loc="left", fontsize=10)
plt.savefig(out_dir + "scaling-0.png")
