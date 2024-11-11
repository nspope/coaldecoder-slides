import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')


out_dir = "../../fig/pop-model/"
if not os.path.exists(out_dir): os.makedirs(out_dir)


def point_array(y, t, n):
    xs = [ t + i for i in range(n)]
    xp = [-t - i for i in range(n)][::-1]
    yy = [y] * n
    return [(x, y) for x, y in zip(xp, yy)], [(x, y) for x, y in zip(xs, yy)]


def choose_parents(children, parents, rng, pr=2):
    idx = []
    p = 0
    for j in range(len(children)):
        p = rng.choice(np.arange(p, min(p + pr, len(parents))))
        idx.append(p)
    out = []
    for c, i in zip(children, idx):
        p = parents[i]
        out.append([(c[0], p[0]), (c[1], p[1])])
    return out

fig, axs = plt.subplots(1, figsize=(4, 3), constrained_layout=True)

time_before = 10
time_after = 10
t_max = time_before + time_after
pop_size = 5
markersize = 3

coords = []
points = {}
for k in range(time_before):
    arr = point_array(t_max - k, 0, pop_size)
    coords.append(arr)
    for pt in arr[0] + arr[1][1:]:
        art, *_ = axs.plot(*pt, marker='o', color='lightgray', markersize=markersize)
        points[pt] = art
for k in range(time_before, time_before + time_after):
    arr = point_array(t_max - k, k - time_before, pop_size)
    coords.append(arr)
    for pt in arr[0] + arr[1]:
        art, *_ = axs.plot(*pt, marker='o', color='lightgray', markersize=markersize)
        points[pt] = art

# boundary
axs.plot((-pop_size, -pop_size), (t_max, t_max - time_before), color="gray")
axs.plot((pop_size, pop_size), (t_max, t_max - time_before), color="gray")
axs.plot((-pop_size, -pop_size - time_before + 1), (t_max - time_before, 1), color="gray")
axs.plot((pop_size, pop_size + time_before - 1), (t_max - time_before, 1), color="gray")
axs.plot((0, -time_before + 2), (t_max - time_before - 1, 1), color="gray")
axs.plot((0, time_before - 2), (t_max - time_before - 1, 1), color="gray")

axs.set_ylabel(r"Time in past $\rightarrow$")
axs.get_xaxis().set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_yticks([])
#axs.get_yaxis().set_visible(False)

# --- just the model ---
for pt in points.values():
    pt.set_color("black")
plt.savefig(out_dir + "pop-model-0.png")

# --- just the mating structure ---

def sample_mating(rng):
    lines = []
    edge_table = []
    for k in range(time_before + time_after, time_before + 1, -1):
        pk, qk = coords[k - 1]
        pkk, qkk = coords[k - 2]
        arr = choose_parents(pk, pkk, rng, pr=2)
        arr += choose_parents(qk, qkk, rng, pr=2)
        for pts in arr:
            ln, *_ = axs.plot(*pts, color='black')
            # store edge
            a, b = pts
            ch = (a[0], b[0])
            pa = (a[1], b[1])
            edge_table.append((ch, pa))
            lines.append(ln)
    
    for k in range(time_before + 1, 1, -1):
        pk, qk = coords[k - 1]
        pkk, qkk = coords[k - 2]
        arr = choose_parents(pk + qk[1:], pkk + qkk[1:], rng, pr=3)
        for pts in arr:
            ln, *_ = axs.plot(*pts, color='black')
            # store edge
            a, b = pts
            ch = (a[0], b[0])
            pa = (a[1], b[1])
            edge_table.append((ch, pa))
            lines.append(ln)

    return lines

lines = sample_mating(np.random.default_rng(200))
plt.savefig(out_dir + "pop-model-1.png")

# --- just the muts

muts = {}
for i, pt in enumerate(coords[-1][0]):
    muts[pt] = "black" #"red" if i < 4 else "black"
for i, pt in enumerate(coords[-1][1]):
    muts[pt] = "red" if i > 0 else "black"

for art in points.values():
    art.set_color("lightgray")
for crd, col in muts.items():
    points[crd].set_color(col)
for art in lines: art.remove()

plt.savefig(out_dir + "pop-model-2.png")

# --- now the genealogy

def sample_tree(rng):
    lines = []
    edge_table = []
    for k in range(time_before + time_after, time_before + 1, -1):
        pk, qk = coords[k - 1]
        pkk, qkk = coords[k - 2]
        arr = choose_parents(pk, pkk, rng, pr=2)
        arr += choose_parents(qk, qkk, rng, pr=2)
        for pts in arr:
            ln, *_ = axs.plot(*pts, color='black')
            # store edge
            a, b = pts
            ch = (a[0], b[0])
            pa = (a[1], b[1])
            edge_table.append((ch, pa))
            lines.append(ln)
    
    for k in range(time_before + 1, 1, -1):
        pk, qk = coords[k - 1]
        pkk, qkk = coords[k - 2]
        arr = choose_parents(pk + qk[1:], pkk + qkk[1:], rng, pr=3)
        for pts in arr:
            ln, *_ = axs.plot(*pts, color='black')
            # store edge
            a, b = pts
            ch = (a[0], b[0])
            pa = (a[1], b[1])
            edge_table.append((ch, pa))
            lines.append(ln)

    colors = {}
    for pt in coords[-1][0]:
        colors[pt] = muts[pt]
    for pt in coords[-1][1]:
        colors[pt] = muts[pt]
    for edge, line in zip(edge_table, lines):
        ci, pi = edge
        if not ci in colors:
            colors[ci] = "lightgray"
        col = colors[ci]
        if not pi in colors:
            colors[pi] = col
        elif colors[pi] == "lightgray":
            colors[pi] = col
        elif colors[pi] == "red" and col == "black":
            colors[pi] = col
        line.set_color(col)
        if col == "lightgray":
            line.set_zorder(0)
        elif col == "red":
            line.set_zorder(1)
        elif col == "black":
            line.set_zorder(2)
    points = []
    for pt, col in colors.items():
        art, *_ = plt.plot(*pt, marker="o", color=col, markersize=markersize)
        points.append(art)

    return lines, points, edge_table

for art in points.values(): art.remove()
lines, points, edge_table = sample_tree(np.random.default_rng(200))

plt.savefig(out_dir + "pop-model-3.png")


# --- now the integral

def update(frame):
    global lines
    global points
    for art in lines: art.remove()
    for art in points: art.remove()
    lines, points, edge_table = sample_tree(np.random.default_rng(frame))

ani = FuncAnimation(fig, update, repeat=True, frames=100, interval=100)
writer = PillowWriter(fps=10)
ani.save(out_dir + "intgr-anim-0.gif", writer=writer, dpi=300)
