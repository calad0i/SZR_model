# Manual for MSZR scripts

The scripts are dependent on the following libraries: `numpy, numba, matplotlib, tqdm`. Please prepare the environment before executing the codes.

## Files

`cluster.py`: Functions for cluster size / critical points searches. In the implementation, the buffers are used for storing the contact lists.

`sorted_list.py`: Utility functions for implementing the data structure of a sorted list. Though the performance is not as good as the native python dictionary, it is compatible with `numba` and outperforms the `numba.typed.Dict` when the size is $\sim 10^4$.

`SZR_contact_tracing`: Functions implementing the MSZR/SZR model with the contact-tracing algorithm. Inferior performance compare to the bond-queue algorithm on single occupancy ($\sim 0.05\times$), but can handle high occupancy as well as the modified SZR model.

`utility.py`: Utility functions for plotting results. Not important.

---

### `sorted_list.py`

All functions in this file are expected to work with a buffer of a 1-dimensional `np.ndarray` of dtype `uint64`. Specifically, the first element should record the number of elements currently stored. This buffer can be re-interpreted as a C-continuous `np.ndarray` of dimension `(N,4)` of dtype uint16. Specifically, the four elements are named `v,u,x,y` from the least significant end of `uint64`. The unused space of the buffer should _always_ erased to 0 to prevent potential issues when calling `index_sorted`. Notice that `(x,y)` are regarded as key and `(v,u)` are regarded as values for this specific application.

---

#### `index_sorted(buf, x, y)`

Returns the index of the query `(x,y)`. If such key does not exist, create it with `(v,u)=(0,0)` and return the index.

`buf`: 1D `np.ndarray` of `uint64`

`x,y`: `uint16`

---

#### `purge_index(buf, idx)`

Remove the `idx`-th element from the buffer `buf`.

`buf`: 1D `np.ndarray` of `uint64`

`idx`: `int`

---

#### `update_sorted(buf, x, y, dv, u)`

Set the value of key `(x,y)` to `(v_old+dv,u)`. However `v_old+dv==0`, remove the key-value pairf from the buffer. Notice that the overflow happened on `v_old+dv` is intended.

`buf`: `np.ndarray` of `uint64`

`x,y,dv,u`: `uint16`

---

#### `set_sorted(buf, x, y, v, u)`

Set the value of key `(x,y)` to `(v,u)`

`buf`: 1D `np.ndarray` of `uint64`

`x,y,dv,u`: `uint16`

---

#### `purge_coord_sorted(buf, x, y)`

Remove the key-value pair with key being `(x,y)` from the buffer, if exists.

`buf`: 1D `np.ndarray` of `uint64`.

`x,y`: `uint16`.

---

### `SZR_contact_tracing.py`

The model uses a `np.ndarray` of dimension `(X,Y,3)` with dtype `uint16` for the lattice. `s_buf` and `z_buf` stores the contact information for grids with non-zero `S` and `Z` respectively. Here, `u` is the number of `S` or `Z` at `x,y`, and `v` is the sum of `Z` or `S` on neighborhoods of `x,y`.

---

#### `update_cell(lattice, x, y, ds, dz, dr, s_buf64, z_buf64)`

Update buffers `s_buf64`, `z_buf64`, and the lattice `lattice`, with changes in components `S,Z,R` being `ds,dz,dr` at position `x,y`.

`lattice`: `np.ndarray` of shape `(X,Y,3)` and `uint16`

`x,y,ds,dz,dr`: `uint16`

`s_buf64,z_buf64`: 1D `np.ndarray` of `uint64`

---

#### `weighted_sample(p)`

Weighted sample, returns natural number `i` with probability proportional to `p[i]`. Note that `p` does not need to be normalized.

`p`: 1D `np.ndarray` of `float64`

---

#### `get_modified_xy(lattice, initiator_xy, target_id)`

Randomly choose a neighborhood of `initiator_xy` with non-zero `X` in `lattice` to be modified. `X` is determined by `target_id`, with `0->S, 1->Z, R->2`. Return the chosen neighbor's coordinate in `tuple(uint16,uint16)`, togete

`lattice`: `np.ndarray` of shape `(X,Y,3)` and `uint16`

`initiator_xy`: `tuple(uint16,uint16)`

`target_id`: `int`

---

#### `(m)szr_sample(lattice, s_buf16, z_buf16, alpha)`

Randomly a grid from the contact lists to start an action (kill or bite), according to the (M)SZR model with `alpha`. Return the initiator's coordinate, changes in `(S,Z,R)`, and time elapsed in this step.

Here, `alpha` is $α=κ/β$. Notice that `s_buf16` and `z_buf16` are re-interprets of `s_buf64` and `z_buf64`, which should point to the same buffer and share modifications.

`lattice`: `np.ndarray` of shape `(X,Y,3)` and `uint16`

`s_buf16,z_buf16`: C-continuous `np.ndarray` of dimension `(N,3)` and dtype `uint16`

`alpha`: `float`

---

#### `run_lattice(lattice, s_buf64, z_buf64, alpha, mszr)`

Simulate the provided `lattice`, with `s_buf64` being the current contact lists `z_buf64`, until reaches stability (contact list emptied). Here, `alpha` is $α=κ/β$. If `mszr` is true, the simulation will comply to the MSZR rule; else, it will comply with the SZR rule.

`lattice`: `np.ndarray` of shape `(X,Y,3)` and `uint16`

`s_buf64,z_buf64`: 1D `np.ndarray` of `uint64`

`alpha`: `float`

`mszr`: `boolean`

---

#### `nb_seed(seed)`

Set the random seed to `seed` inside `numba` for reproducibility.

`seed`: int

---

#### `exec(L, seed, alpha, mszr, occupancy)`

Initialize a `lattice` of shape `(L,L,3)` with `occupancy`, put all `S` on position `(0,0)` to be `Z`, run it until stable (contact list emptied), return the `lattice` after simulation.

When $n=$`occupancy` is an integer ($n-\lfloor n\rfloor=0$), `S`=$n$ is set for all grid points when initialize the `lattice`. If $n$ is not an integer, $p:=n-\lfloor n\rfloor$. For each grid point, put `S`=$\lfloor n\rfloor$ with a probability $p$ and `S`=$\lceil n\rceil$ with a probability $1-p$. If `mszr` is true, the simulation will comply to the MSZR rule; else, it will comply to the SZR rule.

This function will output the peak buffer sizes for `S` and `Z`, also the total time elapsed to `stdout`

`L`: `int`, no greater than 65535 (upper lim of `uint16`)

`seed`: `int`

`alpha`: `float`

`mszr`: `boolean`

`occupancy`: `float`

---

#### `cluster_size(L, seed, alpha, occupancy, mszr)`

Same to `exec`, but return only the cluster size as an integer in the end, and prints nothing to `stdout`.

`L`: `int`, no greater than 65535 (upper lim of `uint16`)

`seed`: `int`

`alpha`: `float`

`mszr`: `boolean`

`occupancy`: `float`

---

#### `batch_clusters_size(L, seed_init, alpha, occupancy, run_number, mszr)`

Run `cluster_size(L, seed, alpha, occupancy, mszr)` in batch with multithreading, with `seed` ranged from `seed_init` to `seed_init+run_number-1` (incl). Totally `run_number` simulations will be done, and the cluster sizes will be returned in a 1D `np.ndarray`.

`L`: `int`, no greater than 65535 (upper lim of `uint16`)

`seed_init`: `int`

`alpha`: `float`

`occupancy`: `float`

`run_number`: `int`

`mszr`: `boolean`

---

#### `get_fit(sizes, const=None)`

Get the slope of the plot of $s^\sigma-P_{s^{2-\tau}\geq s}$ in the plateau part. Param `const` is unused.

`sizes`: 1D `np.ndarray`

`const`: Unused and arbitrary

---

#### `alpha_search(L, seed_init, alpha_low, alpha_high, occupancy, batch, epsilon, max_step, mszr)`

Perform the "quasi" binary search for $α_c$ as described in Section 6, with `batch_clusters_size(L, seed_init, alpha, occupancy, run_number, mszr)` and lower and higher boundaries for $\alpha_c$ being `alpha_low` and `alpha_high`. The search exits if $|slope|<$`epsilon`, $|slope|<3\sigma_{fit}$, or `max_step` is reached. Though, only `epsilon=0` is used in the project.

`L`: `int`, no greater than 65535 (upper lim of `uint16`)

`seed_init`: `int`

`alpha_low,alpha_high`: `float`

`occupancy`: `float`

`batch`: `int`

`epsilon`: `float`

`max_step`: `int`

`mszr`: `boolean`
