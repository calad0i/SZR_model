from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
import numpy as np
from cluster import get_fit


def plt_lattice(lattice: np.ndarray, path: str = None):
    patches = [mpatches.Patch(color=(1, 0, 0), label="Z"),
               mpatches.Patch(color=(0, 1, 0), label="S"),
               mpatches.Patch(color=(0, 0, 0), label="R")
               ]
    lattice_fig = np.empty_like(lattice, dtype=np.float32)
    th = np.max(lattice)
    lattice_fig[:, :, 0] = lattice[:, :, 1]
    lattice_fig[:, :, 1] = lattice[:, :, 0]
    lattice_fig[:, :, 2] = 0*lattice[:, :, 2]
    plt.figure(figsize=(6, 6), dpi=150)
    plt.legend(handles=patches, bbox_to_anchor=(1, 1),
               loc=0, borderaxespad=0., prop={'size': 10})
    plt.imshow(lattice_fig/th, interpolation='None')
    if path is None:
        plt.draw()
    else:
        plt.savefig(path, bbox_inches='tight')


def plt_critical(sizes, path: str = None):
    (k, b), cov = get_fit(sizes)  # type: ignore
    p, s = np.histogram(sizes, bins=100)
    s = np.convolve(s, (0.5, 0.5), 'valid')
    p = p/p.sum()
    p = np.cumsum(p[::-1])[::-1]
    x = np.power(s, 36/91)
    y = p*np.power(s, 187/91-2)
    plt.axes(xlabel=r'$s^\sigma$', ylabel=r'$s^{\tau-2}P_{\geq s}$')
    plt.plot(x, y)
    plt.plot([0, x[-1]], [b, b+k*x[-1]])
    plt.annotate("δ:{:.6f}".format(k), xy=(0, 0))
    plt.draw() if path is None else plt.savefig(path, bbox_inches='tight')
