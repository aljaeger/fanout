import os


# Multithreading Options
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
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
from scipy.interpolate import interp1d

# Use LaTeX for rendering text
matplotlib.rcParams['text.usetex'] = True
options = Options(num_cpus=32)


from useful_functions import get_process_fidelity, sparsify, StateSpace, montecarlo_process_fidelity, basisstate_process_fidelity
from functools import cache


omega_s_max = 20

# Set omega_t to match timing condition
omega_t = 1#np.pi / gate_duration
gate_duration = np.pi
num_phonons = 5
n = 4
kappa = 1

@cache
def theory_infidelity_m(args):  # args=[omega_s]
    ratio = args[0]
    m = args[1]
    omega_s = omega_t / ratio
    b = [fock(2*m+4,i) for i in range(2*m+4)]
            
    H_small = 0* b[1]*b[1].dag() 
    for l in np.arange(0,m+1):
        if m >= l:
            H_small += omega_t * np.sqrt(l+1)* np.sqrt(m-l)* b[2+2*l]*b[0+2*l].dag() 
        if m-1 >= l and l != 0:
            H_small += omega_t * np.sqrt(l)* np.sqrt(m-l)* b[3+2*l]*b[1+2*l].dag() 
        if l != 0:
            H_small += omega_s *np.sqrt(l) * b[1+2*l]*b[0 + 2*l].dag() 

    H_small = H_small + H_small.dag()
    fid = np.abs((b[0].dag() * mesolve(H_small, b[0], np.linspace(0, gate_duration, 200)).states[-1]))**2
    myfidelity = fid 
    return 1-myfidelity


@cache
def find_optimal_omega_s(num_qubits):
    plt.figure(figsize=(8,6))
    omega_t = 1
    gate_duration = np.pi / omega_t

    ns_extended=np.arange(2,31)
    untimed_fidelities = []
    plt.figure(figsize=(8,6))
    plt.xlabel("Ratio $\\Omega_c / \\Omega_t$")
    plt.ylabel("Gate Error")

    fidelity_list = []
    m_list = []

    for m in np.arange(1,num_qubits,1): # loop over the number of ones on the target qubit states. note, the sum goes to num_qubits - 1 = num_target_qubits
        #print("m:",m)
        start_time = time.time()
        
        
        constraint = {'type': 'ineq', 'fun': lambda x: omega_s_max - x[0]}
        
        ratios = np.linspace(0.05, 0.25, 1000).tolist()
        arguments = [(ratio, m) for ratio in ratios]
        
        with multiprocessing.Pool(processes=12) as pool:    
            ratio_fidelities = pool.map(theory_infidelity_m, arguments)
        
        plt.plot(1/np.array(ratios), np.array(ratio_fidelities), alpha=0.3)
        fidelity_list.append(np.array(ratio_fidelities))
        m_list.append(m)
        end_time = time.time()
        #print(f"Time taken: {end_time - start_time:.6f} seconds")
        for k in np.arange(int(1/np.max(ratios) /2), int(1/np.min(ratios) /2)+1):
            plt.axvline(np.sqrt(4*k**2 - 1), color="grey", linestyle="--", alpha=0.2)

    total_fid = sum([comb(np.max(m_list), m) * fid for m, fid in zip(m_list, fidelity_list)]) / 2**np.max(m_list)
    plt.plot(1/np.array(ratios), total_fid, label=f"Average", color="black")

    ratios = np.array(ratios)
    total_fid = np.array(total_fid)
    print(ratios.shape)
    print(total_fid.shape)
    total_fid = total_fid[omega_t/ratios > np.max(omega_t/ratios) - 2]
    ratios = ratios[omega_t/ratios > np.max(omega_t/ratios) - 2]
    omega_s_optimal = omega_t/ratios[np.argmin(total_fid)]

    plt.legend()
    plt.tight_layout()

    
    plt.savefig("graphs2/test_plot_timingcondition2.pdf")

    plt.close()
    return omega_s_optimal
from scipy.sparse import csr_matrix, issparse  

def make_sparse(quantum_object):
    quantum_object_dense = quantum_object.full()  # Get the dense matrix
    quantum_object_dims = quantum_object.dims
    quantum_object_sparse = csr_matrix(quantum_object_dense)  # Convert to sparse matrix

    quantum_object = Qobj(quantum_object_sparse, dims=quantum_object_dims)
    #print(quantum_object.data)
    return quantum_object


#@cache
def compute_fidelity_m(control_state, num_ones, kappa, omega_s, omega_t):
    if control_state == 0:
        p = 1  # 1 phonon if the control state is 0
    else:
        p = 0  # 0 phonons if the control state is 1

    # phonon, number of es, number of fs
    dims = [num_phonons+1, num_ones+1, num_ones+1]
    ## Reduce omega_s to match the timing condition
    ##k_approx = omega_s_max / 2 / omega_t  # approx number of offres excitations
    #k = np.floor(k_approx)  # round down
    #omega_s = 2*k*omega_t #np.sqrt(4*k**2 - n/4) * omega_t

    def Dicke_Hamiltonian():
        vec1 = basis(dims, [0, 0, 0]) 
        vec1 = make_sparse(vec1)
        Hamiltonian = vec1 * vec1.dag() * 0
        # Verify it's sparse
        
        for p in range(num_phonons+1):
            for num_es in range(num_ones+1):
                for num_fs in range(num_ones+1):
                    state1 = [p, num_es, num_fs]
                    state2 = [p, num_es + 1, num_fs]
                    if not all(0 <= state1[i] < dims[i] for i in range(len(state1))) or not all(0 <= state2[i] < dims[i] for i in range(len(state1))):
                        continue


                    vec1 = make_sparse(basis(dims, state1))
                    
                    vec_carrier = make_sparse(basis(dims, state2))

                    if num_ones - num_es - num_fs <= 0:
                        continue
                    multiplier =  np.sqrt((num_es+1) * (num_ones - num_es - num_fs)) 
                    
                    #print("carrier: |", state2, "><", state1, "| : ", multiplier)

                    Hamiltonian += omega_t * multiplier * (vec_carrier*vec1.dag() + vec1*vec_carrier.dag())
        
        for p in np.arange(0,num_phonons+1):
            for num_es in np.arange(0,num_ones+1):
                for num_fs in range(0, num_ones+1):
                    state1 = [p, num_es, num_fs]
                    state2 = [p-1, num_es - 1, num_fs + 1]
                    if not all(0 <= state1[i] < dims[i] for i in range(len(state1))) or not all(0 <= state2[i] < dims[i] for i in range(len(state1))):
                        continue

                    vec1 = make_sparse(basis(dims,state1))

                    vec_sideband = make_sparse(basis(dims, state2) )
                    
                    if num_es - num_fs <= 0:
                        continue
                    multiplier = np.sqrt((num_fs+1)* (num_es - num_fs) * (p))
                    
                    #print("sideband: |", state2, "><", state1,  "|: ", multiplier)

                    Hamiltonian += omega_s * multiplier * (vec_sideband*vec1.dag() + vec1*vec_sideband.dag())

        return Hamiltonian

    def destroy_op():
        vec1 = make_sparse(basis(dims, [0, 0, 0]))
        c_op = vec1 * vec1.dag() * 0

        for p in np.arange(0,num_phonons+1):
            for num_es in np.arange(0,num_ones+1):
                for num_fs in range(0, num_ones+1):
                    state1 = [p, num_es, num_fs]
                    state2 = [p-1, num_es, num_fs]
                    if not all(0 <= state1[i] < dims[i] for i in range(len(state1))) or not all(0 <= state2[i] < dims[i] for i in range(len(state1))):
                        continue
                    
                    vec_n = make_sparse(basis(dims,state1))
                    vec_n_minus1 = make_sparse(basis(dims, state2) )
                    
                    multiplier = np.sqrt(p)
                    c_op +=  multiplier * vec_n_minus1*vec_n.dag()

        return c_op

    from scipy.sparse import issparse, csr_matrix


    
    
    hamiltonian = Dicke_Hamiltonian()
    print(hamiltonian.data)
    #hamiltonian = Qobj(csr_matrix(hamiltonian.data), dims=hamiltonian.dims)
    #print(issparse(hamiltonian.data))

    c_ops = [make_sparse(destroy_op() * np.sqrt(kappa)), make_sparse(destroy_op().dag() * np.sqrt(kappa))]
   # print(hamiltonian)
    for c_op in c_ops:
        print(c_op.data)
    basis_vec = basis(dims, [p, 0, 0])
    #print("dimension of hilbert space:", np.prod(dims))
    #print("now processsing m=", num_ones)
    
    e_ops = [basis_vec*basis_vec.dag()] 
    
    print(np.prod(dims))
    

    if np.prod(dims) < 0:
        result = mesolve(hamiltonian, basis_vec, [0,gate_duration], c_ops=c_ops, e_ops=e_ops)

        fids = result.expect[0]  # Expectation for the first operator (e.g., basis_vec * basis_vec.dag())
    else:
        result = mcsolve(hamiltonian, basis_vec, [0,gate_duration], c_ops=c_ops, ntraj=1000, e_ops=e_ops, options={'progress_bar': False, 'num_cpus': 32})
        fids = result.expect[0]
    return fids[-1]

num_targets = n-1

def compute_fidelity(kappa, omega_s, omega_t, num_targets):
    fidelities = []
    for control_state in [0,1]:
        for num_ones in np.arange(0,num_targets + 1):
            fid = compute_fidelity_m(control_state, num_ones, kappa, omega_s, omega_t)

            fidelities.append(comb(num_targets, num_ones) * fid)
    final_fidelity = sum(fidelities) / 2 ** n
    return final_fidelity

ns = np.arange(2,21)
kappas = np.linspace(0., 0.01, 11)

kappa_grid, n_grid,  = np.meshgrid(kappas, ns)
timed_fidelities = np.zeros_like(n_grid.reshape(-1))
timed_fidelities = list(timed_fidelities)
from tqdm import tqdm

for i, nkappa in enumerate(tqdm(zip(n_grid.reshape(-1), kappa_grid.reshape(-1)), desc="Processing", total=len(n_grid.reshape(-1)))):
    n, kappa = nkappa
    omega_s = find_optimal_omega_s(num_qubits=n)
    print(omega_s)
    
    timed_fidelities[i] = compute_fidelity(kappa,omega_s, omega_t, n-1)
    timed_fidelities_grid = np.array(timed_fidelities).reshape(n_grid.shape)
    np.savez('heating_grid_data.npz', n_grid=n_grid, kappa_grid=kappa_grid, timed_fidelities_grid=timed_fidelities_grid, ns=ns, kappas=kappas)

np.savez('heating_grid_data.npz', n_grid=n_grid, kappa_grid=kappa_grid, timed_fidelities_grid=timed_fidelities_grid, ns=ns, kappas=kappas)
