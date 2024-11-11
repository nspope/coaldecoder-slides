import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

out_dir = "../../fig/arg-3d/"

#fig = plt.figure(constrained_layout=True, figsize=(6,6))
fig = plt.figure(constrained_layout=False, figsize=(6,6))
ax = fig.add_subplot(projection='3d')

xmin = 0.5
xmax = 3.5
ymin = 1
ymax = 4
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(False)
# probably want these with labels
ax.set_axis_off()
ax.set_box_aspect([4, 1, 1])  # Elongate the z-axis
bbox = fig.bbox_inches.from_bounds(0, 0.95, 6, 3.4)


# haplotypes
for yi in range(1, 5):
    ax.plot((xmin, xmax), (yi, yi), zs=(0, 0),  zdir='z', color='gray', linestyle="dashed", dashes=(3, 1), linewidth=2, alpha=0.3)
ax.text(xmin, ymin, z=0, zdir='x', s=r"$0 \leftarrow$", ha='center', va='top')
ax.text(xmax, ymin, z=0, zdir='x', s=r"$\rightarrow L$", ha='center', va='top')
ax.text(xmin/2 + xmax/2, 0.5, z=0, zdir='x', s="Position on sequence", ha='center', va='top')

# ancestral haplotypes
dashes=(2,1)
haplo = []

# first tree
x1 = [1] * 10
y = [1, 2, 3, 4, -1, -1, -1, -1, -1, -1]
y[4] = (y[0] + y[1]) / 2
y[5] = (y[0] + y[1] + y[2]) / 3 
y[6] = (y[0] + y[1] + y[2] + y[3]) / 4
y[7] = y[6]
z = [0, 0, 0, 0, 1, 2, 3, 4, -1, -1, - 1]
l1 = []
edge_table = [(0, 4), (1, 4), (4, 5), (2, 5), (3, 6), (5, 6), (6, 7)]
for i in [4, 5, 6]: # cross tree connections
    ln, *_ = ax.plot((xmin, x1[i]), (y[i], y[i]), zs=(z[i], z[i]), color="black", linestyle="dashed", dashes=dashes)
    ln.set_visible(False)
    haplo.append(ln)
for (i, j) in edge_table: # within tree connections
    line, *_ = ax.plot((x1[i], x1[j]), (y[i], y[j]), zs=(z[i], z[j]),  zdir='z', color='black')
    l1.append(line)
# show haplotypes
for art in l1: art.set_visible(False)
plt.savefig(out_dir + "arg-3d-0.png", bbox_inches=bbox)
for art in l1: art.set_visible(True)
for i in [0, 1, 2, 3, 4, 5, 6]:
    ax.plot(x1[i], y[i], zs=z[i], zdir="z", color='black', marker='o', markersize=3)
ax.text(x1[0], ymin, z=0, zdir='x', s="$x_1$", ha='center', va='top')

plt.savefig(out_dir + "arg-3d-1.png", bbox_inches=bbox)

# second tree
x2 = [2] * 10
y[8] = (y[2] + y[3]) / 2
z[8] = 1
edge_table = [(0, 4), (1, 4), (4, 6), (2, 8), (3, 8), (8, 6), (6, 7)]
l2 = []
for i in [4, 6]: # cross tree connections
    ln, *_ = ax.plot((x1[i], x2[i]), (y[i], y[i]), zs=(z[i], z[i]), color="black", linestyle="dashed", dashes=dashes)
    ln.set_visible(False)
    haplo.append(ln) 
for (i, j) in edge_table: # within tree connections
    line, *_ = ax.plot((x2[i], x2[j]), (y[i], y[j]), zs=(z[i], z[j]),  zdir='z', color='black')
    l2.append(line)
for i in [0, 1, 2, 3, 4, 8, 6]:
    ax.plot(x2[i], y[i], zs=z[i], zdir="z", color='black', marker='o', markersize=3)
ax.text(x2[0], ymin, z=0, zdir='x', s="$x_2$", ha='center', va='top')

# highlight changes
l1[3].set_color("firebrick") # highlight
l2[3].set_color("dodgerblue") # highlight
plt.savefig(out_dir + "arg-3d-2.png", bbox_inches=bbox)
l1[3].set_color("black") # reset
l2[3].set_color("black") # reset

# third tree
x3 = [3] * 10
y[9] = y[6]
z[9] = 2.5
edge_table = [(0, 4), (1, 4), (4, 9), (2, 8), (3, 8), (8, 9), (9, 7)]
l3 = []
for i in [4, 8]: # cross tree connections
    ln, *_ = ax.plot((x2[i], x3[i]), (y[i], y[i]), zs=(z[i], z[i]), color="black", linestyle="dashed", dashes=dashes)
    ln.set_visible(False)
    haplo.append(ln)
for (i, j) in edge_table: # within tree connections
    line, *_ = ax.plot((x3[i], x3[j]), (y[i], y[j]), zs=(z[i], z[j]),  zdir='z', color='black')
    l3.append(line)
for i in [0, 1, 2, 3, 4, 8, 9]:
    ax.plot(x3[i], y[i], zs=z[i], zdir="z", color='black', marker='o', markersize=3)
ax.text(x3[0], ymin, z=0, zdir='x', s="$x_3$", ha='center', va='top')

l2[2].set_color("firebrick") # highlight
l2[5].set_color("firebrick") # highlight
l3[2].set_color("dodgerblue") # highlight
l3[5].set_color("dodgerblue") # highlight
plt.savefig(out_dir + "arg-3d-3.png", bbox_inches=bbox)
l2[2].set_color("black") # reset
l2[5].set_color("black") # reset
l3[2].set_color("black") # reset
l3[5].set_color("black") # reset

for i in [4, 8, 9]: # cross tree connections
    ln, *_ = ax.plot((x3[i], xmax), (y[i], y[i]), zs=(z[i], z[i]), color="black", linestyle="dashed", dashes=dashes)
    ln.set_visible(False)
    haplo.append(ln)

for art in haplo: # highlight
    art.set_color("firebrick")
    art.set_visible(True)

text_label = ax.text(x1[0] / 2 + x2[0] / 2, y[6], z=z[6], s='Ancestral\nhaplotype (node)', size=10, zdir='x', color='firebrick', va='bottom', ha='center')

plt.savefig(out_dir + "arg-3d-4.png", bbox_inches=bbox)

for art in haplo: # reset
    art.set_color("black")
text_label.remove()

# colored edge
xedge = np.array([x1[4], x1[6], x2[6], x2[4]])
yedge = np.array([y[4], y[6], y[6], y[4]])
zedge = np.array([z[4], z[6], z[6], z[4]])
_, zedge = np.meshgrid(yedge, zedge)
xedge, yedge = np.meshgrid(xedge, yedge)
surf = ax.plot_surface(xedge, yedge, zedge, color="firebrick", alpha=0.5)
text_label = ax.text(x1[0] / 2 + x2[0] / 2, y[6], z=z[6], s='Inherited\nhaplotype (edge)', size=10, zdir='x', color='firebrick', va='bottom', ha='center')
plt.savefig(out_dir + "arg-3d-5.png", bbox_inches=bbox)

text_label.remove()
surf.remove()
xedge = np.array([xmin, xmin, xmax, xmax])
yedge = np.array([y[0], y[4], y[4], y[0]])
zedge = np.array([z[0], z[4], z[4], z[0]])
_, zedge = np.meshgrid(yedge, zedge)
xedge, yedge = np.meshgrid(xedge, yedge)
surf = ax.plot_surface(xedge, yedge, zedge, color="firebrick", alpha=0.5)
plt.savefig(out_dir + "arg-3d-6.png", bbox_inches=bbox)
