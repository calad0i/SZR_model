import numpy as np
import numba as nb
from numba import config, njit, prange, threading_layer
from numba import int64
from numba.typed import List
import time

@njit
def check(lattice:np.ndarray, queue):  # obsolete
    
    L, _ = lattice.shape
    
    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    
    counter = 0
    
    xz, yz = np.where(lattice == 1)
    zlist = List()
    
    for i in range(len(xz)):
        zlist.append((xz[i], yz[i]))
    #zlist = np.transpose(np.array(np.where(lattice == 1)))
    
    for xyz in zlist:

        for xyn in left(xyz), right(xyz), up(xyz), down(xyz):
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
    
    if len(queue) == 0:
        return lattice, queue
    
    ind = np.random.randint(len(queue))
    r = np.random.uniform(0, 1)
    
    if r < b/(b + k):
        xyz, xys = queue.pop(ind)
        assert(lattice[xyz] == 1)
        assert(lattice[xys] == 0)
        lattice[xys] = 1
        
        for xyn in left(xys), right(xys), up(xys), down(xys):
            if lattice[xyn] == 0:
                queue.append((xys, xyn))
                
        
        if len(queue) == 0:
            return lattice, queue
        
        for xyn in left(xys), right(xys), up(xys), down(xys):
            if lattice[xyn] == 1 and xyn != xyz:
                queue.remove((xyn, xys))
 
        

    else:
        xyz, xys = queue.pop(ind)
        assert(lattice[xyz] == 1)
        assert(lattice[xys] == 0)
        lattice[xyz] = 2
        
        if len(queue) == 0:
            return lattice, queue
        
        for xyn in left(xyz), right(xyz), up(xyz), down(xyz):
            if lattice[xyn] == 0 and xyn != xys:
                queue.remove((xyz, xyn))
               
    
    return lattice, queue



@njit
def init_lattice(L:int, xy0, sparsity):
    
    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    
    lattice = np.zeros((L, L), dtype = int64)
    randmatx = np.random.uniform(0, 1, size = (L, L))
    lattice = np.where(randmatx < sparsity, 3, 0)  # don't mess up with count
    lattice[xy0] = 1
    queue = List()
    
    for xyn in left(xy0), right(xy0), up(xy0), down(xy0):
        if lattice[xyn] == 0:
            queue.append((xy0, xyn))
    
    return lattice, queue

@njit
def init_lattice_(lattice:np.ndarray, queue:list, xy0):
    
    L, _ = lattice.shape

    left   = lambda xy: ((xy[0]-1)%L, xy[1]      )
    right  = lambda xy: ((xy[0]+1)%L, xy[1]      )
    up     = lambda xy: (      xy[0], (xy[1]+1)%L)
    down   = lambda xy: (      xy[0], (xy[1]-1)%L)
    
    lattice[:, :] = 0
    randmatx = np.random.uniform(0, 1, size = (L, L))
    lattice = np.where(randmatx < sparsity, 3, 0)  # don't mess up with count
    lattice[xy0] = 1
    
    queue = List()
    
    for xyn in left(xy0), right(xy0), up(xy0), down(xy0):
        if lattice[xyn] == 0:
            queue.append((xy0, xyn))

    
    return lattice, queue
    

@njit
def run_lattice_(lattice:np.ndarray, queue:list, k:float, b:float):

    counter = 0    
    while len(queue) != 0:
        counter = counter + 1
        lattice, queue = update(lattice, queue, k, b)
        
#        if counter%100 == 0:
#            print(str(counter) + ":" + str(time.time() - start_time))
            
    return lattice, queue

def run_lattice(lattice:np.ndarray, queue, k:float, b:float):
    start_time = time.time()
    lattice, queue = run_lattice_(lattice, queue, k, b)
    print(time.time() - start_time)
    
    return lattice, queue

  
def run_check_lattice(L:int, xy0, k:float, b:float): # obsolete
    
    lattice, queue = init_lattice(L, xy0)
    counter = 0
    while len(queue) != 0:
        
        if check(lattice, queue) == False:
            print("error")
            break
           
        counter = counter + 1
        lattice, queue = update(lattice, queue, k, b)
        
#        if counter%100 == 0:
#            print(str(counter) + ":" + str(time.time() - start_time))
        
    return lattice, queue

config.THREADING_LAYER = 'threadsafe'
@njit(parallel = True)
def cluster_size(s_list:np.ndarray, lattice:np.ndarray, queue:list, alpha, sparsity):
    N = s_list.shape[0]    
    for i in prange(N):
        lattice, queue = init_lattice(1024, (512, 512), sparsity)
        lattice, queue = run_lattice_(lattice, queue, alpha, 1)
        s_list[i] = np.count_nonzero(lattice == 1) + np.count_nonzero(lattice == 2)
    return s_list

@njit
def nb_seed(seed):
    np.random.seed(seed)
    
def count_cluster(N, L, sparsity, alpha):
    
    s_list = np.zeros(N, dtype = int)
    lattice, queue = init_lattice(L, (int(L/2), int(L/2)), sparsity)
    start_time = time.time()
    s_list = cluster_size(s_list, lattice, queue, alpha, sparsity)
    print("alpha: " + str(alpha))
    print("sparsity:" + str(sparsity))
    print("time: " + str(time.time() - start_time))
    
    np.save("s_list_"+str(alpha) + "_" + str(sparsity)+".npy", s_list, allow_pickle = False)
    return s_list

def slope(N, sparsity, alpha):
    s_list = count_cluster(N, 1024, sparsity, alpha)
    k_list = np.zeros(100) # bootstrap
    for i in range(100):
        s_tmp = np.random.choice(s_list, size = len(s_list), replace = True)
        s, counts = np.unique(s_tmp, return_counts = True)
        s_sig = np.power(s, 36/91)
        s_sig_min, s_sig_max = np.power(3600, 36/91), np.power(40000, 36/91)
        pgts = (np.sum(counts) - np.cumsum(counts) + counts)/np.sum(counts)
        k, b = np.polyfit(s_sig[(s_sig >= s_sig_min)*(s_sig <= s_sig_max)],  
                      (np.power(s, 187/91-2)*pgts)[(s_sig >= s_sig_min)*(s_sig <= s_sig_max)], 1)
        k_list[i] = k
    
    k = np.average(k_list)
    std = np.std(k_list)
    print("slope:" + str(k))
    print("std:" + str(std))
    return k, std

def init_alpha(sparsity, alpha1, alpha2, N): # alpha1 < alpha2
    k1, dk1 = slope(N, sparsity, alpha1)
    k2, dk2 = slope(N, sparsity, alpha2)
    loglist = np.array([[k1, dk1, alpha1], [k2, dk2, alpha2]])
    
    return loglist
    

def new_alpha(loglist, sparsity, N):
        
    k, b = np.polyfit(loglist[:, 2], loglist[:, 0], 1, w = 1/loglist[:, 1])
    alpha3 = -b/k
    
    k3, dk3 = slope(N, sparsity, alpha3)
    ind = np.searchsorted(loglist[:, 2], alpha3)
    loglist = np.insert(loglist, ind, 0, axis = 0)
    loglist[ind,:] = k3, dk3, alpha3
    
    if dk3 > np.abs(k3):
        return loglist, ind, False # increase N if allowed
    else:
        return loglist, ind, True  # continue
    
def find_alpha(sparsity, alpha1, alpha2, N_max):
    N = 1000
    
    loglist = init_alpha(sparsity, alpha1, alpha2, N)
    
    loglist, ind, N_good = new_alpha(loglist, sparsity, N)
    
    while N <= N_max:
        while N_good == True:
            loglist, ind, N_good = new_alpha(loglist, sparsity, N)
            
        N = N*4
        loglist, ind, N_good = new_alpha(loglist, sparsity, N)
        
    return loglist, ind

    
nb_seed(524149)

sparsity_list = [0.1, 0.2, 0.3, 0.35, 0.38, 0.4]
alpha1_list = [0.355, 0.255, 0.143, 0.079, 0.0375, 0.009]
alpha2_list = [0.356, 0.26, 0.144, 0.08, 0.0395, 0.01]
alphac_list = []
N_max = 16000

for i in range(len(sparsity_list)):
    loglist, ind = find_alpha(sparsity_list[i], alpha1_list[i], alpha2_list[i], N_max)
    alphac_list.append(loglist[ind, 2])    
    np.savetxt("loglist_" + str(sparsity_list[i]) + ".txt", loglist)

np.save("alphac_list.npy", alphac_list, allow_pickle = False)