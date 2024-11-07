import tskit
import os
import numpy as np
import itertools
import matplotlib


def draw_clade(axs, x, y, color='black', linestyle="-", fade=False):
    x0, x1 = x
    y0, y1 = y
    xn = (x0 + x1) / 2
    line_kwargs = {"color" : color, "linewidth" : 1, "linestyle" : linestyle}
    patch_kwargs = {"linewidth" : 0, "color" : "gray", "alpha" : 0.3}
    if fade:
        line_kwargs["alpha"] = 0.1
        line_kwargs["linestyle"] = "--"
        patch_kwargs["alpha"] = 0.03
    px = [x0, xn, x1]
    py = [y0, y1, y0]
    axs.add_patch(matplotlib.patches.Polygon(list(zip(px,py)), **patch_kwargs))
    axs.plot((x0, x1), (y0, y0), **line_kwargs)
    axs.plot((x0, xn), (y0, y1), **line_kwargs)
    axs.plot((x1, xn), (y0, y1), **line_kwargs)


def draw_trio(axs, x, y, clade_width, before_recomb=True, color='black', edge_highlight='black', highlight_path=False, fade_outgroup=False):
    line_kwargs = {"color" : color, "linewidth" : 1}
    highlight_kwargs = {"color" : edge_highlight, "linewidth" : 1, "linestyle" : "-"}
    path_kwargs = {"color" : (edge_highlight if highlight_path else color), "linewidth" : (2 if highlight_path else 1)}
    if edge_highlight != "black":
        highlight_kwargs["linewidth"] = 2
        highlight_kwargs["linestyle"] = "--"
        highlight_kwargs["dashes"] = (3, 1)
    x1, x2, x3, x4, x5 = np.linspace(x[0], x[1], 5)
    y0, y1, y2, y3, y4, y5 = np.linspace(y[0], y[1], 6)
    y4 = y5
    draw_clade(axs, (x1 - clade_width / 2, x1 + clade_width / 2), (y0, y1), color=color)
    draw_clade(axs, (x2 - clade_width / 2, x2 + clade_width / 2), (y0, y1), color=color)
    draw_clade(axs, (x3 - clade_width / 2, x3 + clade_width / 2), (y0, y1), color=color)
    draw_clade(axs, (x4 - clade_width / 2, x4 + clade_width / 2), (y0, y3), color=color, fade=fade_outgroup)
    x12 = (x1 + x2) / 2
    x23 = (x2 + x3) / 2
    x123 = x2
    x1234 = x3
    axs.plot((x1, x12), (y1, y2), **line_kwargs)
    if before_recomb:
        axs.plot((x2, x12), (y1, y2), **highlight_kwargs)
    else:
        axs.plot((x2, x23), (y1, y2), **highlight_kwargs)
    axs.plot((x3, x23), (y1, y2), **line_kwargs)
    axs.plot((x12, x123), (y2, y3), **(path_kwargs if before_recomb else line_kwargs))
    axs.plot((x23, x123), (y2, y3), **(line_kwargs if before_recomb else path_kwargs))
    axs.plot((x123, x1234), (y3, y4), **path_kwargs)
    axs.plot((x4, x1234), (y3, y4), **line_kwargs)
    # nodes
    axs.plot(x1, y1, marker="o", color="black", markersize=4)
    axs.plot(x2, y1, marker="o", color="black", markersize=4)
    axs.plot(x3, y1, marker="o", color="black", markersize=4)
    axs.plot(x12, y2, marker="o", color="black", markersize=4)
    axs.plot(x23, y2, marker="o", color="black", markersize=4)
    axs.plot(x123, y3, marker="o", color="black", markersize=4)
    axs.plot(x4, y3, marker="o", color="black", markersize=4)
    axs.plot(x1234, y4, marker="o", color="black", markersize=4)
    return (
        np.array([x1, y1]), 
        np.array([x2, y1]), 
        np.array([x3, y1]), 
        np.array([x12, y2]), 
        np.array([x23, y2]), 
        np.array([x123, y3]), 
        np.array([x4, y3]), 
        np.array([x1234, y4])
    )

