import numpy as np
from numba import njit, prange
from SZR_contact_tracing import nb_seed, update_cell, szr_sample, mszr_sample


@njit
def cluster_size(L: int, seed: int, alpha: float = 0.25, occupancy: float = 1, mszr: bool = True):
    # Initialize a lattice, run it, and return the cluster size.
    nb_seed(L)
    lattice = np.zeros((L, L, 3), dtype=np.uint16)
    v = int(np.floor(occupancy))
    lattice[:, :, 0] = v
    lattice[:, :, 0] += (np.random.rand(L, L) < occupancy-v)
    nb_seed(seed)
    occupancy = int(np.ceil(occupancy))

    s_buf64 = np.zeros((L*10*occupancy), dtype=np.uint64)
    z_buf64 = np.zeros((L*10*occupancy), dtype=np.uint64)
    s_buf16 = np.frombuffer(s_buf64, dtype=np.uint16).reshape(-1, 4)
    z_buf16 = np.frombuffer(z_buf64, dtype=np.uint16).reshape(-1, 4)

    update_cell(lattice, np.uint16(0), np.uint16(
        0), np.uint16(-lattice[0, 0, 0]), np.uint16(lattice[0, 0, 0]), np.uint16(0), s_buf64, z_buf64)

    for i in range(np.sum(lattice[:, :, :2])*2):
        if s_buf64[0] != 0 and z_buf64[0] != 0:
            if mszr is True:
                x, y, (ds, dz, dr), dt = mszr_sample(
                    lattice, s_buf16, z_buf16, alpha
                )
            else:
                x, y, (ds, dz, dr), dt = szr_sample(
                    lattice, s_buf16, z_buf16, alpha
                )
            update_cell(lattice, x, y, ds, dz, dr, s_buf64, z_buf64)
        else:
            assert z_buf64[0] == 0 and s_buf64[0] == 0
            break
    return np.sum(lattice[:, :, 1:])


@njit(parallel=True)  # type: ignore
def batch_clusters_size(L: int, seed_init: int, alpha: float = 0.25, occupancy: int = 1, run_number: int = 128, mszr: bool = True):
    # Run multiple simulations of lattices with the same setting in parallel, and return all cluster sizes.
    res_container = np.empty(run_number, dtype=np.int64)
    for i in prange(run_number):
        res_container[i] = cluster_size(
            L, seed_init+i, alpha=alpha, occupancy=occupancy, mszr=mszr)
    return res_container


def get_fit(sizes: np.ndarray, const=None):
    # Get the best line fit for the $s^{\tau-2}P_{\ge s}$ - $s^\sigma$ in the plateau region, return fitted params and covariance matrix.
    n, s = np.histogram(sizes, bins=100)
    s = np.convolve(s, (0.5, 0.5), 'valid')
    p = n/n.sum()
    p[::-1] = np.cumsum(p[::-1])
    x = np.power(s, 36/91)
    y = p*np.power(s, 187/91-2)
    w = np.power(s, 187/91-2)*np.power(n*(n.sum()-n)+1, -0.5)
    best_cov = np.array([[np.inf, np.inf], [np.inf, np.inf]])
    best_fit = np.array([-1, -1])
    l = np.max(x)
    fit = np.nan
    for low in np.linspace(int(0.1*l), int(0.28*l)):
        for high in np.linspace(int(0.6*l), int(0.9*l)):
            mask = (high >= x) & (x >= low)
            if np.sum(mask) < 10:
                continue
            try:
                fit, cov = np.polyfit(
                    x[mask], y[mask], w=w[mask], deg=1, cov=True)
            except:
                cov = np.array([[np.inf, np.inf], [np.inf, np.inf]])
            if cov[0, 0] < best_cov[0, 0]:
                best_cov = cov
                best_fit = fit
    return best_fit, best_cov


def alpha_search(L: int, seed_init: int, alpha_low: float, alpha_high: float, occupancy=1, batch=1000, epsilon=0., max_step=20, mszr=True):
    # Perform critical point search in binary way, must specifying the upper and lower values. However, if the slope changed in an unexpected way, the last update in the boundary in the opposite direction will be undone (e.g. decrease α -> slope decreases -> last lower bond update will be undone.).
    const = 2-187/91
    last_delta = np.inf
    last_alpha_low = alpha_low
    last_alpha_high = alpha_high
    his = np.empty((max_step, 2))
    for i in range(max_step):
        print('[{:.5f},{:.5f}]'.format(alpha_low, alpha_high))
        alpha = 0.5*(alpha_high+alpha_low)
        sizes = batch_clusters_size(
            L, seed_init, alpha=alpha, occupancy=occupancy, run_number=batch, mszr=mszr)
        fit, cov = get_fit(sizes, const)
        delta = fit[0]  # type:ignore
        sigma = np.sqrt(cov[0, 0])
        thres = max(epsilon, 2.5*sigma)
        his[i] = alpha, delta
        if delta > thres:
            last_alpha_low = alpha_low
            alpha_low = alpha
            if delta > last_delta and last_delta > 0:
                alpha_high = last_alpha_high
            last_delta = delta
            print(alpha, delta)
        elif delta < -thres:
            last_alpha_high = alpha_high
            alpha_high = alpha
            if delta < last_delta and last_delta < 0:
                alpha_low = last_alpha_low
            last_delta = delta
            print(alpha, delta)
        else:
            print(alpha, delta)
            return alpha, (alpha_high-alpha), fit, sizes, his[:i+1], cov
    return alpha, (alpha_high-alpha), fit, sizes, his[:i+1], cov  # type:ignore
