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
options = Options(num_cpus=5)



from useful_functions import get_process_fidelity, sparsify, StateSpace, montecarlo_process_fidelity, basisstate_process_fidelity


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
# Matplotlib Plot Formatting
font = {'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)

# Experimental Params 
max_sideband = 2*np.pi*120*0.4
gate_duration = 0.1

# Define state space
s = StateSpace(1,3, num_states=4, num_phonons=4)
a = s.a
zero, one, e, f = s.internal_states[:4]

# Define fanout operation
fanout_transition = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.one*s.one.dag()} | {i: s.zero*s.zero.dag() - s.one*s.one.dag() for i in s.target_indices})
fanout_idle = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.zero*s.zero.dag()} | {i: s.zero* s.zero.dag() + s.one*s.one.dag() for i in s.target_indices})
fanout = fanout_idle + fanout_transition

basis_transform = s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons,0).dag(), 1: e*zero.dag()}) + s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons,0).dag(), 1: one*one.dag()})

# Define Hamiltonian
omega_s = max_sideband
ratio =(gate_duration*omega_s/(np.pi) - 1)**-1

intermediate_ratio = np.sqrt(ratio)*omega_s
omega_d = ratio * omega_s

k_approx = max_sideband / 2 / omega_d
k = np.ceil(k_approx)
ratio_new = 1/2/k
omega_s = ratio / ratio_new * omega_s 
ratio = ratio_new

H = lambda omega_s, ratio:  omega_s * sum([s.local({0: a, j: f*e.dag()}) for j in s.target_indices]
                           + [ratio * s.local({j: e*one.dag()}) for j in s.target_indices])

hamiltonian = H(omega_s, ratio)
hamiltonian += hamiltonian.dag()

plt.figure(figsize=(9,10))

fidelity = np.zeros(200,dtype=np.complex128)
total_weight = 0
times = np.linspace(0,gate_duration, 200)

hamiltonian = H(omega_s, ratio)
hamiltonian += hamiltonian.dag()
    
for basis_digit, initial_state in zip(s.basis_digits_reduced, s.basis_states_reduced):
    initial_state = basis_transform*initial_state
    
    linestyle="-"
    print(basis_digit)
    if basis_digit[1] == 1:
        pass
    target_phase = (-1)**np.sum(basis_digit[2:])
    if basis_digit[1] == 0:
        target_phase = 1
        linestyle="-."
    
    states = mesolve(hamiltonian, initial_state, np.linspace(0, gate_duration-np.pi/max_sideband,200), progress_bar=None).states
        
    
    amplitudes = np.array([(initial_state.dag()*state) for state in states])
    phases = np.arctan2((amplitudes * target_phase).imag, (amplitudes * target_phase).real) / np.pi 

    excitations = sum([s.local({j: e*e.dag()}) for j in s.target_indices])
    label = "|"
    label+=str(basis_digit)[3:-1]
    label+="\\rangle"
    label = label.replace(",","")
    label = label.replace(" ","")
    label = "$"+label+"$"
    plt.subplot(3,1,1)
    plt.plot(np.array(times) / times[-1], phases, label=label, linestyle=linestyle)
    plt.subplot(3,1,2)
    plt.plot(np.array(times) / times[-1], np.abs(amplitudes), label=label, linestyle=linestyle)
    
    
    weight = comb(s.target_qubits, sum(basis_digit[2:]))
    total_weight += weight
    fidelity += amplitudes * target_phase * weight
    print(amplitudes[-1], target_phase)
print(fidelity.shape)
fidelity = abs(fidelity / total_weight)
plt.subplot(3,1,1)
plt.ylabel("Phase (units of $\pi$)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -2.65), shadow=True, ncol=4)  # Adjust the values of bbox_to_anchor as needed

#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplot(3,1,2)
plt.ylabel("Target State Overlap")
plt.grid()
plt.subplot(3,1,3)
plt.ylabel("Fidelity")
plt.plot(np.array(times) / times[-1], fidelity)
plt.grid()
plt.xlabel("Gate Evolution")

plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.17, wspace=0.1, hspace=0.2)

print("Final Fidelity:", fidelity[-1])
plt.savefig("figures/basis_state_evolution.pdf")


