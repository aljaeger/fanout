import os
# Multithreading Options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import threadpoolctl
threadpoolctl.threadpool_limits(1)

import progressbar
import numpy as np
import scipy as sp
from qutip import *
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from math import comb
import time
import pickle
from matplotlib.colors import LogNorm 
import matplotlib
# Use LaTeX for rendering text
matplotlib.rcParams['text.usetex'] = True
from useful_functions import get_process_fidelity, sparsify, StateSpace, montecarlo_process_fidelity, basisstate_process_fidelity
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
# Matplotlib Plot Formatting
font = {'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)


from scipy.sparse import csr_matrix, issparse  


def make_sparse(quantum_object):
    quantum_object_dense = quantum_object.full()  # Get the dense matrix
    quantum_object_dims = quantum_object.dims
    quantum_object_sparse = csr_matrix(quantum_object_dense)  # Convert to sparse matrix

    quantum_object = Qobj(quantum_object_sparse, dims=quantum_object_dims)
    #print(quantum_object.data)
    return quantum_object

#Experimental Params 
max_sideband = 2*np.pi*120*0.4
gate_duration = np.pi

from scipy.optimize import minimize

# test theoretical model:
n=5
omega_s = 100
omega_t = 10

s = StateSpace(1,n-1, num_states=4, num_phonons=4)
a = s.a
zero, one, e, f = s.internal_states[:4]

H = lambda omega_s, ratio:  omega_s * sum([s.local({0: a, j: f*e.dag()}) for j in s.target_indices]
                               + [ratio * s.local({j: e*one.dag()}) for j in s.target_indices])

hamiltonian = H(omega_s, omega_t / omega_s)
hamiltonian += hamiltonian.dag()
basis_transform = s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons,0).dag(), 1: e*zero.dag()}) \
+ s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons,0).dag(), 1: one*one.dag()}) / 2
basis_transform = basis_transform + basis_transform.dag()


for basis_digit, initial_state in zip(s.basis_digits_reduced, s.basis_states_reduced):
    if basis_digit[1] == 1:
        continue
    initial_state = basis_transform * initial_state
    initial_state = make_sparse(initial_state)
    hamiltonian = make_sparse(hamiltonian)
    final_states = mesolve(hamiltonian, initial_state, np.linspace(0, gate_duration,200), progress_bar=None).states
    fids = [np.abs((initial_state.dag()*final_state))**2 for final_state in final_states]
    plt.plot( np.linspace(0, gate_duration,200), fids)



b = [fock(100,i) for i in range(100)]
myobjective = 0
for m in np.arange(0,n)[::-1]:
    H_small = 0* b[1]*b[1].dag() 
    for l in np.arange(0,10):
        if m >= l:
            H_small += omega_t * np.sqrt(l+1)* np.sqrt(m-l)* b[2+2*l]*b[0+2*l].dag() 
        if m-1 >= l and l != 0:
            H_small += omega_t * np.sqrt(l)* np.sqrt(m-l)* b[3+2*l]*b[1+2*l].dag() 
        if l != 0:
            H_small += omega_s *np.sqrt(l) * b[1+2*l]*b[0 + 2*l].dag() 

    H_small = H_small + H_small.dag()
    
    initial_state= b[0]
    
    final_states =  mesolve(H_small, initial_state, np.linspace(0, gate_duration, 200)).states
    fids = [np.abs((initial_state.dag()*final_state))**2 for final_state in final_states]
    
    plt.plot(np.linspace(0, gate_duration,200), fids, linestyle="--", color="grey")
plt.show()
myobjective = myobjective / 2**(n-1)


fids = []
ms = []
controls = []



avg_fidelity = sum([comb(np.float128(n)-1, np.float128(m)) * fid for m, fid in zip(ms, fids)]) / 2**np.float128(n)

n_fidelities = []

# Set omega_t to match timing condition
gate_duration = np.pi

max_sideband = 20
omega_t = 1

print(gate_duration)
ns_extended=np.arange(2,101)
untimed_fidelities = []

os.makedirs("figures/timing-selection", exist_ok=True)

for n in ns_extended:
    print("n:",n)
    start_time = time.time()

    omega_s = max_sideband
    
    # Set omega_t to match timing condition
    omega_t = np.pi / gate_duration

    def theory_infidelity(args):  # args=[omega_s]
        
        omega_s = args[0]
        b = [fock(2*n+4,i) for i in range(2*n+4)]
        
        myfidelity = 0
        for m in np.arange(0,n):
            H_small = 0* b[1]*b[1].dag() 
            for l in np.arange(0,n):
                if m >= l:
                    H_small += omega_t * np.sqrt(l+1)* np.sqrt(m-l)* b[2+2*l]*b[0+2*l].dag() 
                if m-1 >= l and l != 0:
                    H_small += omega_t * np.sqrt(l)* np.sqrt(m-l)* b[3+2*l]*b[1+2*l].dag() 
                if l != 0:
                    H_small += omega_s *np.sqrt(l) * b[1+2*l]*b[0 + 2*l].dag() 

            H_small = H_small + H_small.dag()
            H_small = make_sparse(H_small)
            
            fid = np.abs((b[0].dag() * mesolve(H_small, make_sparse(b[0]), np.linspace(0, gate_duration, 200)).states[-1]))**2
            myfidelity += fid * comb(n-1,m)
            
        myfidelity = myfidelity / np.float128(2)**np.float128(n) + 1/2
        return 1-myfidelity
    
    constraint = {'type': 'ineq', 'fun': lambda x: max_sideband - x[0]}
    
    
    from scipy.optimize import differential_evolution
    #result = differential_evolution(theory_infidelity, [(max_sideband-2*omega_t*1.1, max_sideband)], workers=10)
    #omega_s = result.x[0]
    
    omega_s_guesses = np.linspace(max_sideband-2*omega_t*1.1, max_sideband, 100).tolist()
    
    def worker_function(omega_s):
        return theory_infidelity([omega_s])
    num_processes = 32
    with multiprocessing.Pool(processes=num_processes) as pool:    
        omega_s_fidelities = pool.map(worker_function, omega_s_guesses)
    
    untimed_fidelities.append(omega_s_fidelities) 
    omega_s = omega_s_guesses[np.argmin(omega_s_fidelities)]
    plt.plot(omega_s_guesses, omega_s_fidelities)
    plt.axvline(omega_s)
    plt.savefig(f"figures/timing-selection/plot_{n}.pdf")
    plt.close()
    
    myfidelity = theory_infidelity([omega_s])
    n_fidelities.append(myfidelity)
    print(omega_s, max_sideband)
    print("fidelity", 1-myfidelity)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.6f} seconds")

    with open(f"data/qubitscalingsimulation_smallH{int(gate_duration*1000)}.p", "wb") as file:
        pickle.dump([gate_duration, ns_extended, np.array(untimed_fidelities), n_fidelities], file)