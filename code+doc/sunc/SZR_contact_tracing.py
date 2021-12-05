from sorted_list import *
import numpy as np
from numba import njit, prange
from tqdm import trange


@njit
def update_cell(lattice: np.ndarray, x: np.uint16, y: np.uint16, ds: np.uint16, dz: np.uint16, dr: np.uint16, s_buf64: np.ndarray, z_buf64: np.ndarray):
    # Update the state of grid point (x,y) and all of its neighbors in contact lists (s_buf and z_buf).
    s, z, r = lattice[x, y]
    lattice[x, y, 0] += ds
    lattice[x, y, 1] += dz
    lattice[x, y, 2] += dr
    X, Y, _ = lattice.shape
    neighbors_xy = np.array(
        [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int32)
    neighbors_xy[:, 0] = (neighbors_xy[:, 0]+x) % X
    neighbors_xy[:, 1] = (neighbors_xy[:, 1]+y) % Y
    nzs = np.uint16(0)
    nss = np.uint16(0)
    if ds != 0:
        for nx, ny in neighbors_xy:
            nz = lattice[nx, ny, 1]
            nzs += nz
            if nz != 0 and (nx != x or ny != y):  # neighbor have Z!=0
                update_sorted(z_buf64, nx, ny, np.uint16(ds), nz)
        set_sorted(s_buf64, x, y, nzs, np.uint16(s+ds))
    elif s != 0:
        update_sorted(s_buf64, x, y, np.uint16(dz), np.uint16(s+ds))
    if dz != 0:
        for nx, ny in neighbors_xy:
            ns = lattice[nx, ny, 0]
            nss += ns
            if ns != 0 and (nx != x or ny != y):  # neighbor have S!=0
                update_sorted(s_buf64, nx, ny, np.uint16(dz), ns)
        set_sorted(z_buf64, x, y, nss, np.uint16(z+dz))
    elif z != 0:
        update_sorted(z_buf64, x, y, np.uint16(ds), np.uint16(z+dz))


@njit
def weighted_sample(p: np.ndarray):
    # numba implement of weighted sample.
    cumulative_distribution = np.cumsum(p)
    a = np.float32(cumulative_distribution[-1])
    return np.searchsorted(cumulative_distribution, np.random.uniform(0, a), side="right")


@njit
def get_modified_xy(lattice: np.ndarray, initiator_xy: np.ndarray, target_idx: np.uint16):
    # Randomly sample a neighborhood of initiator_xy to be modified in the consequent state updating.
    x, y = initiator_xy
    X, Y, _ = lattice.shape
    neighbors_xy = np.array(
        [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int32)
    neighbors_xy[:, 0] = (neighbors_xy[:, 0]+x) % X
    neighbors_xy[:, 1] = (neighbors_xy[:, 1]+y) % Y
    p = np.empty(neighbors_xy.shape[0], dtype=np.uint16)
    for i in range(neighbors_xy.shape[0]):
        nx, ny = neighbors_xy[i]
        p[i] = lattice[nx, ny, target_idx]
    if np.sum(p) == 0:
        shape = lattice[x-np.uint(2):x+np.uint(3),
                        y-np.uint(2):y+np.uint(3), 0].shape
        ll = np.empty(shape)
        print(x, y)
        ll[:, :] = lattice[x-np.uint(2):x+np.uint(3),
                           y-np.uint(2):y+np.uint(3), 0]
        ll[:, :] -= lattice[x-np.uint(2):x+np.uint(3),
                            y-np.uint(2):y+np.uint(3), 1]
        print(ll, target_idx)
        assert False
    return neighbors_xy[weighted_sample(p)]


@njit
def mszr_sample(lattice: np.ndarray, s_buf16: np.ndarray, z_buf16: np.ndarray, alpha: float = 0.25):
    # Sample a initiator based on the Modified SZR model.
    b = 1
    k = b*alpha
    K = k*s_buf16[0, 0]
    B = b*z_buf16[0, 0]
    dt = np.random.exponential(1/(K+B))
    if np.random.rand() < K/(K+B):
        buf16 = s_buf16[1:s_buf16[0, 0]+np.uint16(1)]
        dszr = (np.uint16(0), np.uint16(-1), np.uint16(1))
        idx = 1
    else:
        buf16 = z_buf16[1:z_buf16[0, 0]+np.uint16(1)]
        idx = 0
        dszr = (np.uint16(-1), np.uint16(1), np.uint16(0))

    initiator_idx = weighted_sample(buf16[:, 1])
    initiator_xy = buf16[initiator_idx, 2:]
    x, y = get_modified_xy(lattice, initiator_xy, np.uint16(idx))

    return x, y, dszr, dt


@njit
def szr_sample(lattice: np.ndarray, s_buf16: np.ndarray, z_buf16: np.ndarray, alpha: float = 0.25):
    # Sample a initiator based on the SZR model.
    b = 1
    k = b*alpha
    ls = s_buf16[0, 0]+np.uint(1)
    lz = z_buf16[0, 0]+np.uint(1)
    ps = np.sum(s_buf16[1:ls, 0]*s_buf16[1:ls, 1])
    pz = np.sum(z_buf16[1:lz, 0]*z_buf16[1:lz, 1])
    if ps != pz:
        print(s_buf16[1:ls])
        print(z_buf16[1:lz])
        print(ps, pz)
        print(lattice[:, :, 0])
        print(lattice[:, :, 1])
        assert False
    K = k*ps
    B = b*pz
    dt = np.random.exponential(1/(K+B))
    if np.random.rand() < K/(K+B):
        buf16 = s_buf16[1:s_buf16[0, 0]+np.uint16(1)]
        dszr = (np.uint16(0), np.uint16(-1), np.uint16(1))
        idx = 1
    else:
        buf16 = z_buf16[1:z_buf16[0, 0]+np.uint16(1)]
        idx = 0
        dszr = (np.uint16(-1), np.uint16(1), np.uint16(0))

    initiator_idx = weighted_sample(buf16[:, 1]*buf16[:, 0])
    initiator_xy = buf16[initiator_idx, 2:]
    x, y = get_modified_xy(lattice, initiator_xy, np.uint16(idx))

    return x, y, dszr, dt


def run_lattice(lattice: np.ndarray, s_buf64: np.ndarray, z_buf64: np.ndarray, alpha: float, mszr: bool = True):
    # Run a lattice until stable, with progress bar.
    s_buf16 = np.frombuffer(s_buf64, dtype=np.uint16).reshape(-1, 4)
    z_buf16 = np.frombuffer(z_buf64, dtype=np.uint16).reshape(-1, 4)
    ms = 0
    mz = 0
    T = 0.
    if mszr:
        sample = mszr_sample
    else:
        sample = szr_sample
    for i in trange(np.sum(lattice[:, :, :2])*2):
        if s_buf64[0] > ms:
            ms = s_buf64[0]
        if z_buf64[0] > mz:
            mz = z_buf64[0]
        if s_buf64[0] != 0 and z_buf64[0] != 0:
            x, y, (ds, dz, dr), dt = sample(lattice, s_buf16, z_buf16, alpha)
            T += dt
            update_cell(lattice, x, y, ds, dz, dr, s_buf64, z_buf64)
        else:
            print('max S, Z buf sizes:', ms, mz)
            print('Time elapsed:', T)
            assert z_buf64[0] == 0 and s_buf64[0] == 0
            break


@njit
def nb_seed(seed):
    # Set random seed for numba functions.
    np.random.seed(seed)


def exec(L: int, seed: int, alpha: float = 0.25, mszr: bool = True, occupancy: int = 1):
    # Initialize a lattice as specified, and run it until stable.
    nb_seed(seed)
    lattice = np.zeros((L, L, 3), dtype=np.uint16)
    lattice[:, :, 0] = occupancy
    s_buf64 = np.zeros((L*10*occupancy), dtype=np.uint64)
    z_buf64 = np.zeros((L*10*occupancy), dtype=np.uint64)

    update_cell(lattice, np.uint16(0), np.uint16(0), np.uint16(-occupancy),
                np.uint16(occupancy), np.uint16(0), s_buf64, z_buf64)
    run_lattice(lattice, s_buf64, z_buf64, alpha, mszr)
    return lattice
