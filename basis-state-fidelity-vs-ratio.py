import os


# Multithreading Options
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
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

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
# Matplotlib Plot Formatting
font = {'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)
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


def plot_basis_state_fidelities(num_qubits):
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

        label = None
        if n < 5:
            label = f"$m={m}$"
        plt.plot(1/np.array(ratios), np.array(ratio_fidelities), alpha=0.3, label=label)
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
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"figures/basis-state-fidelity-{num_qubits}.pdf")

    plt.close()
    return 



ns = np.arange(2,11)

from tqdm import tqdm
for i,n in enumerate(tqdm(ns, desc="Processing", total=len(ns))):
    plot_basis_state_fidelities(num_qubits=n)
    