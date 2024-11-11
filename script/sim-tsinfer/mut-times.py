import numpy as np
import tskit
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

from core import *

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# ----------- #
output_dir = "../../fig/sim-tsinfer/"
if not os.path.exists(output_dir): os.makedirs(output_dir)


#--------- true mutation ages

chunksize = 10e6
inp_dir = "/sietch_colab/natep/trio-coal/sims/osclogalt-big-regmu/"
true_ts = tskit.load(inp_dir + f"osclogaltbig_1.ts")
true_ts = true_ts.trim()
chunks = np.linspace(0, true_ts.sequence_length, int(true_ts.sequence_length / chunksize) + 1)
true_ts = true_ts.keep_intervals([[chunks[0], chunks[1]]]).trim()
infr_ts = tskit.load(inp_dir + f"infer_osclogaltbig_1.0.trees")

import tsdate.evaluation
true, infr = tsdate.evaluation.mutations_time(true_ts, infr_ts, what="child")

drop = np.logical_or(true == 0.0, infr == 0.0)
fig, axs = plt.subplots(1, figsize=(4, 4))
axs.hexbin(true[~drop], infr[~drop], xscale='log', yscale='log', mincnt=1)
axs.axline((10, 10), (100, 100), color="firebrick", linestyle="dashed")
axs.set_xlabel(r"True node age (below mutations)")
axs.set_ylabel(r"Inferred node age (below mutations)")
axs.set_title("tsinfer 0.3.3 + tsdate 0.1.6dev\n10 Mb, 400 diploids, $\mu/r = 2$")
fig.tight_layout()
plt.savefig(output_dir + "mut-times-0.png")

axs.text(0.1, 0.6, "Homoskedastic errors\non log scale", transform=axs.transAxes)
plt.savefig(output_dir + "mut-times-1.png")
