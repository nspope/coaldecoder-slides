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

# ------- #

force_overwrite = False
output_dir = "../../fig/ne-duality/"
if not os.path.exists(output_dir): os.makedirs(output_dir)

time, ICR, CDF, Ne = get_pop_size_and_coal_rate()
ts = get_ts()

x_lower = 0
x_upper = time[-1]


# ------ population size ------ #

fig, axs = plt.subplots(figsize=(6, 3))

axs.set_xlim(x_lower, x_upper)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)

axs.set_ylim(0, 4e4)
axs.set_xlabel(r"Time in past $(t)$")
axs.set_yticks([0, 10000, 20000, 30000, 40000])
axs.set_ylabel(r"Haploid population size $n(t)$")
axs.plot(time, 2 * Ne, color='black', linewidth=1)
label = axs.text(0.41, 0.8, r"$n(t) = 2 N(t)$", color="black", transform=axs.transAxes, fontsize=12)

fig.tight_layout()
plt.savefig(output_dir + "size-to-coal-0.png")


# ----- coalescence rate --------- #

for art in list(axs.lines): art.remove()
label.remove()

axs.set_yticks([0, 0.0001])
axs.set_ylim(0, 1 / 7e3)
axs.set_ylabel(r"Pair coalescence rate $c(t)$")

axs.plot(time, ICR, color='black', linewidth=1)
label = axs.text(0.41, 0.8, r"$c(t) = n(t)^{-1}$", color="black", transform=axs.transAxes, fontsize=12)

plt.savefig(output_dir + "size-to-coal-1.png")


# ----- cdf --------- #

for art in list(axs.lines): art.remove()
label.remove()

axs.set_yticks([0, 0.5, 1])
axs.set_ylim(0, 1)
axs.set_ylabel(r"Pair coalescence $\mathrm{CDF}(t)$")

axs.plot(time, CDF, color='black', linewidth=1)
label = (
r"\["
r"\begin{aligned}"
r"\mathrm{Coalescent} & \implies \mathrm{CDF}(t) = 1 - \exp\{-\int_0^t c(t) dt\} \\"
r"\end{aligned}"
r"\]"
)
label = axs.text(0.3, 0.55, label, color="black", transform=axs.transAxes, fontsize=12, ha='left', va='top')

plt.savefig(output_dir + "size-to-coal-2.png")

label.remove()
label = (
r"\["
r"\begin{aligned}"
r"\mathrm{Coalescent} & \implies \mathrm{CDF}(t) = 1 - \exp\{-\int_0^t c(t) dt\} \\"
r"& \implies c(t) = \frac{\mathrm{CDF}'(t)}{1 - \mathrm{CDF}(t)}"
r"\end{aligned}"
r"\]"
)
label = axs.text(0.3, 0.55, label, color="black", transform=axs.transAxes, fontsize=12, ha='left', va='top')

plt.savefig(output_dir + "size-to-coal-3.png")

# ----- cdf + ecdf --------- #

label.remove()

ecdf_time, ecdf = average_ecdf(ts)

plt.step(ecdf_time, ecdf, where='post', color='firebrick', alpha=0.75)
cdf_label = axs.text(0.43, 0.71, "$\mathrm{CDF}(t)$", color="black", transform=axs.transAxes, fontsize=12, ha='right', va='bottom')
ecdf_label = axs.text(0.47, 0.65, "$\mathrm{ECDF}_{\mathcal{S}}(t)$", color="firebrick", transform=axs.transAxes, fontsize=12, ha='left', va='top')

plt.savefig(output_dir + "size-to-coal-4.png")


# ----- cdf + ecdf + box --------- #

rect = matplotlib.patches.Rectangle((4e4, 0), 5e3, 1.0, color='gray', alpha=0.3, lw=0)
axs.add_patch(rect)
intv_label = axs.text(4.25e4, 0.5, r"$[a, b)$", ha='center', va='center', fontsize=12)

plt.savefig(output_dir + "size-to-coal-5.png")


# ----- ne estimator ------------ #

for art in list(axs.lines): art.remove()
cdf_label.remove()
ecdf_label.remove()
intv_label.remove()
rect.remove()

grid = np.linspace(0, 5e4, 21)
ne_est = 1. / get_ne_est(ts, grid)

axs.set_ylim(0, np.max(ne_est) + 1e3)
axs.set_xlabel(r"Time in past $(t)$")
axs.set_yticks([0, 10000, 20000, 30000, 40000])
axs.set_ylabel(r"Haploid population size $n(t)$")
axs.plot(time, 2 * Ne, color='black', linewidth=1)
true_label = axs.text(0.97, 0.55, r"$n(t)$", color="black", transform=axs.transAxes, ha='right', fontsize=12)

plt.savefig(output_dir + "size-to-coal-6.png")

axs.step(grid, np.append(ne_est, ne_est[-1]), where="post", color='firebrick', linewidth=1, linestyle="--")
est_label = axs.text(0.42, 0.87, r"$\hat{n}(t)$ from $\mathrm{ECDF}_{\mathcal{S}}$", color="firebrick", transform=axs.transAxes, fontsize=12, ha='left')

plt.savefig(output_dir + "size-to-coal-7.png")
