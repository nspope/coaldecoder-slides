import numpy as np

from core import *

#ts_list = [
#    fetch(i, j) for ...

import matplotlib

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

import matplotlib.pyplot as plt

out_dir = "../../fig/roadmap/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

pdf_store = "tmp.pdf.p"
if not os.path.exists(pdf_store):
    pdf_list = get_pdf_msprime()
    pickle.dump(pdf_list, open(pdf_store, "wb"))
else:
    pdf_list = pickle.load(open(pdf_store, "rb"))

#pdf_store2 = "tmp.pdf2.p"
#if not os.path.exists(pdf_store2) or True:
#    pdf_list2 = get_pdf_msprime_unfitted()
#    pickle.dump(pdf_list2, open(pdf_store2, "wb"))
#else:
#    pdf_list2 = pickle.load(open(pdf_store2, "rb"))

tot = 0
ecdf = None
simp_ecdf = None
for i in range(1):
    for j in range(1, 2):
        ts = fetch(i, j)
        sample_sets = [np.flatnonzero(ts.nodes_population == 0), np.flatnonzero(ts.nodes_population == 1)]
        time_grid = np.linspace(0, 6e4, 100)
        tmp = ts.pair_coalescence_counts(sample_sets, indexes=[[0,0],[0,1],[1,1]], time_windows=time_grid, pair_normalise=True, span_normalise=False)
        if ecdf is None:
            ecdf = tmp
        else:
            ecdf += tmp
        node_bin = np.digitize(ts.nodes_time[ts.nodes_time > 0], time_grid) - 1
        node_bin = node_bin[node_bin < time_grid.size - 1]
        node_tot = np.bincount(node_bin, minlength=time_grid.size - 1).astype(np.float64)
        if simp_ecdf is None:
            simp_ecdf = node_tot
        else:
            simp_ecdf += node_tot
        tot += ts.sequence_length

ecdf /= tot
simp_ecdf /= np.sum(simp_ecdf)

fig, axs = plt.subplots(1, figsize=(3.5, 3))

colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[i % len(colors)] for i in range(3)]
pdfs = []
bars = []
br0 = axs.bar(time_grid[:-1], simp_ecdf / np.diff(time_grid), width=np.diff(time_grid), color="gray")
for i, (mid, mass, haz) in enumerate(pdf_list):
    mid = np.append(mid, [np.max(mid), np.min(mid)])
    mass = np.append(mass, [0, 0])
    epdf = ecdf[i] / np.diff(time_grid)
    #xy = np.stack([mid, mass]).T
    #patch = matplotlib.patches.Polygon(xy, facecolor=colors[i], edgecolor=colors[i], alpha=0.2)
    #axs.add_patch(patch)
    br = axs.bar(time_grid[:-1], epdf, width=np.diff(time_grid), color=colors[i], alpha=0.5)
    ln, *_ = axs.plot(mid, mass, color=colors[i])
    bars.append(br)
    pdfs.append(ln)

axs.set_ylim(0, 2e-4)
axs.set_xlim(0, mid.max())
axs.set_ylabel("Probability mass")
axs.set_xlabel("Time to coalescence")
axs.get_xaxis().set_ticks([])
axs.get_yaxis().set_ticks([])
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
fig.tight_layout()

for ln in pdfs: ln.set_visible(False)
for br in bars: 
    for b in br:
        b.set_visible(False)
plt.savefig(out_dir + "/roadmap-0.png")

for b in br0:
    b.set_visible(False)
for br in bars: 
    for b in br:
        b.set_visible(True)
plt.savefig(out_dir + "/roadmap-1.png")

for ln in pdfs: ln.set_visible(True)
plt.savefig(out_dir + "/roadmap-2.png")
