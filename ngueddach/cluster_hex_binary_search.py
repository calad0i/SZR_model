import numpy as np
import numba as nb
from numba import config, njit, prange, threading_layer
from numba import int64
from numba.typed import List
import time

@njit
def check(lattice:np.ndarray, queue):
    
    L, _ = lattice.shape
    
    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    diag_top  = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] + 1)%L)
    diag_down = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] - 1)%L)
    
    counter = 0
    
    xz, yz = np.where(lattice == 1)
    zlist = List()
    
    for i in range(len(xz)):
        zlist.append((xz[i], yz[i]))
    #zlist = np.transpose(np.array(np.where(lattice == 1)))
    
    for xyz in zlist:

        for xyn in left(xyz), right(xyz), up(xyz), down(xyz), diag_top(xyz), diag_down(xyz):
            if lattice[xyn] == 0:
                counter = counter + 1
                if (xyz, xyn) not in queue:
                    print(xyz, xyn)
                    return False
                
    if counter == len(queue):
        return True
    else:
        print(counter, len(queue))
        return False
    
@njit
def update(lattice:np.ndarray, queue:list, k:float, b:float):
    
    L, _ = lattice.shape
    
    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    diag_top  = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] + 1)%L)
    diag_down = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] - 1)%L)
    
    if len(queue) == 0:
        return lattice, queue
    
    ind = np.random.randint(len(queue))
    r = np.random.uniform(0, 1)
    
    if r < b/(b + k):
        xyz, xys = queue.pop(ind)
        assert(lattice[xyz] == 1)
        assert(lattice[xys] == 0)
        lattice[xys] = 1
        
        for xyn in left(xys), right(xys), up(xys), down(xys), diag_top(xys), diag_down(xys):
            if lattice[xyn] == 0:
                queue.append((xys, xyn))
                
        
        if len(queue) == 0:
            return lattice, queue
        
        for xyn in left(xys), right(xys), up(xys), down(xys), diag_top(xys), diag_down(xys):
            if lattice[xyn] == 1 and xyn != xyz:
                queue.remove((xyn, xys))
 
        

    else:
        xyz, xys = queue.pop(ind)
        assert(lattice[xyz] == 1)
        assert(lattice[xys] == 0)
        lattice[xyz] = 2
        
        if len(queue) == 0:
            return lattice, queue
        
        for xyn in left(xyz), right(xyz), up(xyz), down(xyz), diag_top(xyz), diag_down(xyz):
            if lattice[xyn] == 0 and xyn != xys:
                queue.remove((xyz, xyn))
               
    
    return lattice, queue



@njit
def init_lattice(L:int, xy0):
    
    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    diag_top  = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] + 1)%L)
    diag_down = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] - 1)%L)
    
    lattice = np.zeros((L, L), dtype = int64)
    lattice[xy0] = 1
    queue = List()
    queue.append((xy0,  left(xy0)))
    queue.append((xy0, right(xy0)))
    queue.append((xy0,    up(xy0)))
    queue.append((xy0,  down(xy0)))
    queue.append((xy0, diag_top(xy0)))
    queue.append((xy0, diag_down(xy0)))
    
    return lattice, queue

@njit
def init_lattice_(lattice:np.ndarray, queue:list, xy0):
    
    L, _ = lattice.shape

    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    diag_top  = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] + 1)%L)
    diag_down = lambda xy: ((xy[0] + 1 - 2*(xy[1]%2))%L, (xy[1] - 1)%L)
    
    lattice[:, :] = 0
    lattice[xy0] = 1
    
    queue = List()
    queue.append((xy0,  left(xy0)))
    queue.append((xy0, right(xy0)))
    queue.append((xy0,    up(xy0)))
    queue.append((xy0,  down(xy0)))
    queue.append((xy0, diag_top(xy0)))
    queue.append((xy0, diag_down(xy0)))

    
    return lattice, queue
    

@njit
def run_lattice_(lattice:np.ndarray, queue:list, k:float, b:float):

    counter = 0    
    while len(queue) != 0:
        counter = counter + 1
        lattice, queue = update(lattice, queue, k, b)
            
    return lattice, queue

def run_lattice(lattice:np.ndarray, queue, k:float, b:float):
    start_time = time.time()
    lattice, queue = run_lattice_(lattice, queue, k, b)
    print(time.time() - start_time)
    
    return lattice, queue

  
def run_check_lattice(L:int, xy0, k:float, b:float):
    
    lattice, queue = init_lattice(L, xy0)
    counter = 0
    while len(queue) != 0:
        
        if check(lattice, queue) == False:
            print("error")
            break
           
        counter = counter + 1
        lattice, queue = update(lattice, queue, k, b)
                
    return lattice, queue

N=100000
L=512
L2=L//2

config.THREADING_LAYER = 'threadsafe'
@njit(parallel = True)
def cluster_size(s_list:np.ndarray, lattice:np.ndarray, queue:list, alpha:float):
    N = s_list.shape[0]    
    for i in prange(N):
        lattice, queue = init_lattice(L, (L2, L2))
        lattice, queue = run_lattice_(lattice, queue, alpha, 1)
        s_list[i] = np.count_nonzero(lattice)
    return s_list

@njit
def nb_seed(seed):
    np.random.seed(seed)
    
nb_seed(524149)

lo = 0.
hi = 100.

for i in range(10):
    mid = (lo+hi)/2.

    s_list = np.zeros(N, dtype = int)
    print(f"Iteration {i}, initializing lattice...")
    lattice, queue = init_lattice(L, (L2, L2))
    start_time = time.time()
    s_list = cluster_size(s_list, lattice, queue, mid)

    s, counts = np.unique(s_list, return_counts = True)
    s_sig = np.power(s, 36/91)
    pgts = (np.sum(counts) - np.cumsum(counts) + counts)/np.sum(counts)
    k, b = np.polyfit(s_sig[(s_sig >= 50)*(s_sig <= 100)],  (np.power(s, 187/91-2)*pgts)[(s_sig >= 50)*(s_sig <= 100)], 1)

    if k < 0:
        hi = mid
    else:
        lo = mid

    print("Finished current iteration")
    print("Runtime:", time.time() - start_time)
    print("Used alpha:", mid)

np.save(f"s_list_{N}_{L}_{mid}.npy", s_list, allow_pickle = False)