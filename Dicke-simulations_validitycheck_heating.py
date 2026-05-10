import os


# Multithreading Options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
# Matplotlib Plot Formatting
font = {'weight' : 'normal', 'size'   : 18}
plt.rc('font', **font)

from useful_functions import get_process_fidelity, sparsify, StateSpace, montecarlo_process_fidelity, basisstate_process_fidelity
from scipy.sparse import csr_matrix, issparse  

def make_sparse(quantum_object):
    quantum_object_dense = quantum_object.full()  # Get the dense matrix
    quantum_object_dims = quantum_object.dims
    quantum_object_sparse = csr_matrix(quantum_object_dense)  # Convert to sparse matrix

    quantum_object = Qobj(quantum_object_sparse, dims=quantum_object_dims)
    #print(quantum_object.data)
    return quantum_object


# Set omega_t to match timing condition
omega_t = 1#np.pi / gate_duration
omega_s = 20
gate_duration = np.pi
num_phonons = 4
gamma = 0.01
n = 4
print(n, "qubit fanout simulation")
s = StateSpace(1,n-1, num_states=4, num_phonons=num_phonons+1)
a = s.a
zero, one, e, f = s.internal_states[:4]

H = lambda omega_s, omega_t:  sum([omega_s * s.local({0: a, j: f*e.dag()}) for j in s.target_indices]
                               + [omega_t * s.local({j: e*one.dag()}) for j in s.target_indices])

hamiltonian = H(omega_s, omega_t)
hamiltonian += hamiltonian.dag()
hamiltonian = make_sparse(hamiltonian)
c_ops = [s.local({0: a})*np.sqrt(gamma), s.local({0: a.dag()})*np.sqrt(gamma) ]
for i in range(len(c_ops)):
    c_ops[i] = make_sparse(c_ops[i])

basis_transform = s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons,0).dag(), 1: e*zero.dag()}) \
+ s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons,0).dag(), 1: one*one.dag()}) / 2
basis_transform = basis_transform + basis_transform.dag()

label = "Full Simulation"
for basis_digit, initial_state in zip(s.basis_digits_reduced, s.basis_states_reduced):
    if basis_digit[1] == 0:  # apply pi pulse of control qubit is 0.
        initial_state = basis_transform * initial_state
    initial_state = make_sparse(initial_state)
    print(expect(s.local({0: a.dag()*a}), initial_state))
    final_states = mesolve(hamiltonian, initial_state, np.linspace(0, gate_duration,200), c_ops=c_ops, progress_bar=None).states
    fids = [(initial_state*initial_state.dag() *final_state).tr() for final_state in final_states]
    plt.plot( np.linspace(0, 1,200), fids, alpha=0.5, color="black", label=label)
    label = None

num_targets = n-1
fidelities = [[], []]
label = "Block-Diagonal Simulation"
#"""
for control_state in [0, 1]:
    for num_ones in range(num_targets + 1):
        print("Now processing approximation for m=",num_ones, "out of", num_targets + 1)
        # phonon, number of es, number of fs
        dims = [num_phonons+1, num_ones+1, num_ones+1]

        ## Reduce omega_s to match the timing condition
        ##k_approx = max_sideband / 2 / omega_t  # approx number of offres excitations
        #k = np.floor(k_approx)  # round down
        #omega_s = 2*k*omega_t #np.sqrt(4*k**2 - n/4) * omega_t

        def Dicke_Hamiltonian():
            vec1 = basis(dims, [0, 0, 0]) 
            Hamiltonian = vec1 * vec1.dag() * 0
            for p in range(num_phonons+1):
                for num_es in range(num_ones+1):
                    for num_fs in range(num_ones+1):
                        state1 = [p, num_es, num_fs]
                        state2 = [p, num_es + 1, num_fs]
                        if not all(0 <= state1[i] < dims[i] for i in range(len(state1))) or not all(0 <= state2[i] < dims[i] for i in range(len(state1))):
                            continue


                        vec1 = basis(dims, state1)
                        
                        vec_carrier = basis(dims, state2)
                        multiplier =  np.sqrt((num_es+1) * (num_ones - num_es - num_fs)) 
                        if num_ones - num_es - num_fs <= 0:
                            continue
                        print("carrier: |", state2, "><", state1, "| : ", multiplier)

                        Hamiltonian += omega_t * multiplier * (vec_carrier*vec1.dag() + vec1*vec_carrier.dag())
            
            for p in np.arange(0,num_phonons+1):
                for num_es in np.arange(0,num_ones+1):
                    for num_fs in range(0, num_ones+1):
                        state1 = [p, num_es, num_fs]
                        state2 = [p-1, num_es - 1, num_fs + 1]
                        if not all(0 <= state1[i] < dims[i] for i in range(len(state1))) or not all(0 <= state2[i] < dims[i] for i in range(len(state1))):
                            continue

                        vec1 = basis(dims,state1)

                        vec_sideband = basis(dims, state2) 
                        multiplier = np.sqrt((num_fs+1)* (num_es - num_fs) * (p))
                        if num_es - num_fs <= 0:
                            continue
                        print("sideband: |", state2, "><", state1,  "|: ", multiplier)

                        Hamiltonian += omega_s * multiplier * (vec_sideband*vec1.dag() + vec1*vec_sideband.dag())

            return Hamiltonian

        
        
        def creation_op():
            vec1 = basis(dims, [0, 0, 0]) 
            c_op = vec1 * vec1.dag() * 0

            for p in np.arange(0,num_phonons+1):
                for num_es in np.arange(0,num_ones+1):
                    for num_fs in range(0, num_ones+1):
                        state1 = [p, num_es, num_fs]
                        state2 = [p+1, num_es, num_fs]
                        if not all(0 <= state1[i] < dims[i] for i in range(len(state1))) or not all(0 <= state2[i] < dims[i] for i in range(len(state1))):
                            continue
                        
                        vec_n = basis(dims,state1)
                        vec_n_minus1 = basis(dims, state2) 
                        
                        multiplier = np.sqrt(p+1)
                        c_op +=  multiplier * vec_n_minus1*vec_n.dag()

            return c_op

        hamiltonian = Dicke_Hamiltonian()
        c_ops = [creation_op() * np.sqrt(gamma), creation_op().dag() * np.sqrt(gamma)]
    # print(hamiltonian)

        basis_vec = basis(dims, [1-control_state, 0, 0])

        states = mesolve(hamiltonian, basis_vec, np.linspace(0, gate_duration, 200), c_ops=c_ops).states

        fids = [(basis_vec*basis_vec.dag() *state).tr() for state in states]

        if control_state == 0:
            fidelities[0].append(np.array(fids))
        elif control_state == 1:
            fidelities[1].append(np.array(fids))


        plt.plot(np.linspace(0, 1,200), fids, linestyle=":", color="blue", label=label)
        label = None

plt.legend()
plt.xlabel("Gate Evolution")
plt.ylabel("Fidelity")
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.subplots_adjust(left=0.12)

plt.savefig(f"figures/test_{n}.pdf")
plt.close()
#"""

#* EVOLUTION OF RANDOM STATES
s = StateSpace(1,n-1, num_states=4, num_phonons=num_phonons+1)
a = s.a
zero, one, e, f = s.internal_states[:4]

H = lambda omega_s, omega_t:  sum([omega_s * s.local({0: a, j: f*e.dag()}) for j in s.target_indices]
                               + [omega_t * s.local({j: e*one.dag()}) for j in s.target_indices])

hamiltonian = H(omega_s, omega_t)
hamiltonian += hamiltonian.dag()
hamiltonian = make_sparse(hamiltonian)
c_ops = [s.local({0: a})*np.sqrt(gamma), s.local({0: a.dag()})*np.sqrt(gamma) ]

for i in range(len(c_ops)):
    c_ops[i] = make_sparse(c_ops[i])

basis_transform = s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons,0).dag(), 1: e*zero.dag()}) \
+ s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons,0).dag(), 1: one*one.dag()}) / 2
basis_transform = basis_transform + basis_transform.dag()
basis_transform = make_sparse(basis_transform)
num_trials = 100

approx_fids = []
exact_fids = []
for i in range(num_trials):
    print("Now processing state", i, "out of", num_trials)
    if basis_digit[1] == 1:
        continue
    amplitudes = np.random.normal(0,1,size=2**n) 
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    phases = np.exp(1j * np.random.uniform(0,2*np.pi,size=2**n))
    initial_state = sum([amplitude * phase * state for amplitude, phase, state in zip(amplitudes, phases, s.basis_states)])
    initial_state = initial_state 
    initial_state = make_sparse(initial_state)

    fanout_transition = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.one*s.one.dag()} | {i: s.zero*s.zero.dag() - s.one*s.one.dag() for i in s.target_indices})
    fanout_idle = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.zero*s.zero.dag()} | {i: s.zero* s.zero.dag() + s.one*s.one.dag() for i in s.target_indices})
    fanout = fanout_idle + fanout_transition
    fanout = make_sparse(fanout)
    target_state = fanout * initial_state
    target_state = basis_transform * target_state
    target_state = make_sparse(target_state)
    initial_state = basis_transform * initial_state
    initial_state = make_sparse(initial_state)
    
    
    print(expect(s.local({0: a.dag()*a}), initial_state))
    final_states = mesolve(hamiltonian, initial_state, np.linspace(0, gate_duration,200), c_ops=c_ops, progress_bar=None).states
    fids = [(target_state*target_state.dag() *final_state).tr() for final_state in final_states]

    exact_fids.append(np.abs(fids[-1]))

    print("Exact Simulation      :", fids[-1])
    
    #plt.plot( np.linspace(0, gate_duration,200), fids)

    # Fidelity under the simplification of block-diagonality
    num_ones_array = [np.sum(digits[2:]) for digits in s.basis_digits]
    control_states_array = [digits[1] for digits in s.basis_digits]
    total_fidelity = sum([fidelities[control_states_array[i]][num_ones_array[i]] * amplitudes[i]**2 for i in range(len(amplitudes))]) 
    approx_fids.append(np.abs(total_fidelity[-1]))
    
    #plt.plot(np.linspace(0, gate_duration,200), total_fidelity, linestyle="--")
    print("Block-diag Simulation :", total_fidelity[-1])

    if i % 10 == 0:
        print(exact_fids, approx_fids)
        plt.grid()
        plt.plot([min(np.min(approx_fids), np.min(exact_fids)), max(np.max(approx_fids), np.max(exact_fids))],
                 [min(np.min(approx_fids), np.min(exact_fids)), max(np.max(approx_fids), np.max(exact_fids))],
                 color="blue", alpha=0.5, linestyle="--", label="Ideal Approximation")
        plt.scatter(exact_fids, approx_fids, color="red", label="Random State Simulation")
        
        plt.xlabel("Exact Fidelity")
        plt.ylabel("Basis State Approximation" )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/test_heating_scatterplot_{n}.pdf")

        plt.close()
    
        # Compute RMSE
        rmse = np.sqrt(np.mean((np.array(exact_fids) - np.array(approx_fids)) ** 2))
        print("RMSE:", rmse)


print(exact_fids, approx_fids)
plt.grid()
plt.plot([min(np.min(approx_fids), np.min(exact_fids)), max(np.max(approx_fids), np.max(exact_fids))],
            [min(np.min(approx_fids), np.min(exact_fids)), max(np.max(approx_fids), np.max(exact_fids))],
            color="blue", alpha=0.5, linestyle="--", label="Ideal Approximation")
plt.scatter(exact_fids, approx_fids, color="red", label="Random State Simulation")

plt.xlabel("Exact Fidelity")
plt.ylabel("Basis State Approximation" )
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/test_heating_scatterplot_{n}.pdf")

plt.close()

# Compute RMSE
rmse = np.sqrt(np.mean((np.array(approx_fids) - np.array(exact_fids)) ** 2))
print("RMSE:", rmse)
