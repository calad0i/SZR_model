import numpy as np
from numba import njit

# Utility functions for sorted list for data storage: {(x,y)->(v,u)}, with all of x,y,v,u being uint16. The sorted lists are 2 dimensional and C-continuous. For input into these functions, re-interpert the array as a 1 dimensional uint64 vector. The first element is used to store the size of current buffer, which should never exceed the buffer's length minus 1.


@njit
def index_sorted(buf: np.ndarray, x: np.uint16, y: np.uint16):
    # Return the index of the key (x,y). If no such key exists, create it and return the index.
    k = (np.uint32(x) << 32)+(np.uint32(y) << 48)
    idx = np.searchsorted(buf[1:buf[0]+np.uint64(1)], k)+1
    mask32_64 = np.uint64(18446744069414584320)
    if buf[idx] & mask32_64 == k and buf[0]+1 > idx:
        return idx
    else:
        assert buf.shape[0] > buf[0]+1
        buf[0] += 1
        buf[idx+1:buf[0]+np.uint64(1)] = buf[idx:buf[0]]
        buf[idx] = k
        return idx


@njit
def purge_index(buf: np.ndarray, idx: np.int32):
    # Remove index idx from the buffer.
    buf[idx:buf[0]] = buf[idx+1:buf[0]+np.uint64(1)]
    buf[buf[0]] = 0
    buf[0] -= 1


@njit
def update_sorted(buf: np.ndarray, x: np.uint16, y: np.uint16, dv: np.uint16, u: np.uint16):
    # Set value of key (x,y) to be (v+dv,u). If v is 0 after update, remove the key-value pair.
    idx = index_sorted(buf, x, y)
    mask0_15 = np.uint64(65535)
    mask32_64 = np.uint64(18446744069414584320)
    buf[idx] = ((mask32_64 & buf[idx])
                + np.uint64(np.uint32(u) << 16)
                + (mask0_15 & np.uint64(dv+buf[idx])))
    if ((mask0_15 & buf[idx])) == 0:
        purge_index(buf, idx)


@njit
def set_sorted(buf: np.ndarray, x: np.uint16, y: np.uint16, v: np.uint16, u: np.uint16):
    # Set value of key (x,y) to be (v,u).
    idx = index_sorted(buf, x, y)
    if v == 0 or u == 0:
        purge_coord_sorted(buf, x, y)
        return
    buf[idx] = (np.uint64(v)
                + (np.uint32(u) << 16)
                + (np.uint32(x) << 32)
                + (np.uint32(y) << 48))


@njit
# Remove key (x,y) from the buffer.
def purge_coord_sorted(buf: np.ndarray, x: np.uint16, y: np.uint16):
    k = (np.uint32(x) << 32)+(np.uint32(y) << 48)
    idx = np.searchsorted(buf[1:buf[0]+np.uint64(1)], k)+1
    mask32_64 = np.uint64(18446744069414584320)
    if (buf[idx] & mask32_64) == k:
        purge_index(buf, idx)
