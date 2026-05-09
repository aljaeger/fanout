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

#Experimental Params 
max_sideband = 2*np.pi*120*0.4
gate_duration = 0.1


omega_s = max_sideband

s = StateSpace(1,2, num_states=4, num_phonons=4)
a = s.a
zero, one, e, f = s.internal_states[:4]

fanout_transition = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.one*s.one.dag()} | {i: s.zero*s.zero.dag() - s.one*s.one.dag() for i in s.target_indices})
fanout_idle = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.zero*s.zero.dag()} | {i: s.zero* s.zero.dag() + s.one*s.one.dag() for i in s.target_indices})
fanout = fanout_idle + fanout_transition

basis_transform = s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons,0).dag(), 1: e*zero.dag()}) + s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons,0).dag(), 1: one*one.dag()})

def H(omega_s, ratio):
    return [max_sideband * s.local({0: a.dag(), 1: e*zero.dag()}),
                            omega_s * sum([s.local({0: a, j: f*e.dag()}) for j in s.target_indices]
                           + [ratio * s.local({j: e*one.dag()}) for j in s.target_indices]),
                                -max_sideband * s.local({0: a.dag(), 1: e*zero.dag()})]

    
def worker_function(t):
    #print(t, end=" ")
    ratio = (t*omega_s/(np.pi) - 1)**-1
    
    execution_times = [np.pi/2/max_sideband, t - np.pi/max_sideband, np.pi/2/max_sideband]

    hamiltonians = H(omega_s, ratio)
    hamiltonians = [h+h.dag() for h in hamiltonians]
    
    # Compute fidelity
    return (get_process_fidelity(hamiltonians, execution_times, fanout, collapse_operators=[]))

fidelities = []
times = list(np.linspace(np.pi/max_sideband+0.01, 0.3,301))


for k in np.arange(1,100):
    ratio1 = 1/2/k
    t = np.pi/omega_s*(1/ratio1 + 1) 
    if t < np.max(times) and t > np.min(times):
        times.append(t)

times.sort()
times = np.array(times)
ratios = np.array([(t*omega_s/(np.pi) - 1)**-1 for t in times])


# Compute fidelity with multiprocessing
bar = progressbar.ProgressBar(maxval=len(times), 
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' | ', progressbar.Timer(), ' | ', progressbar.ETA()])
fidelities = []
with multiprocessing.Pool(processes=1) as pool:    
    result = pool.imap(worker_function, times)
    
    # Update the progress bar as tasks complete
    bar.start()
    for i, fid in enumerate(result):
        bar.update(i+1)
        fidelities.append(fid)
    bar.finish()


plt.figure(figsize=(10,6))

label = "Timing Condition"
for k in np.arange(1,30):
    t = 2*np.pi*k/omega_s + np.pi/max_sideband
    if t < np.max(times) and t > np.min(times):
        color="black"
        plt.plot([t, t], [np.min(1-np.array(fidelities))/2, np.max(1-np.array(fidelities))*2], color=color, linestyle="--", alpha=0.5, label=label)
        label = None  # turn off label for remaining lines
plt.xlabel("Gate Duration (ms)")
plt.ylabel("Infidelity")

theoretical_infidelities = s.target_qubits * np.array(ratios) ** 2
timed_theoretical_infidelities = np.pi**2/16*(s.target_qubits)*(s.target_qubits-1)*(np.array(times)*omega_s/(np.pi) - 1)**-4
x= (np.array(times)*omega_s/(np.pi) - 1)**-1
k=5
temp = sum([comb(s.target_qubits, m)*4*m*x**2/(1+m*x**2)*np.sin((np.array(times) - np.pi/omega_s)/2 * omega_s*np.sqrt((1+m*x**2)))**2 for m in range(s.num_qubits)])/2**s.num_qubits


plt.semilogy(times, temp, label="Theory",color="green", linestyle="-.", alpha=0.5)
plt.semilogy(times, 1-np.array(fidelities), label="Simulation")
plt.plot(times, theoretical_infidelities, label="Upper Bound")

plt.grid(axis="y")
plt.ylim(np.min(1-np.array(fidelities))/2, np.max(1-np.array(fidelities))*2)

plt.plot()
#omega_d = np.pi / np.array(times)
plt.legend()

plt.savefig("figures/fidelity-vs-time.pdf")
plt.close()

def H(omega_s, ratio):
    return [max_sideband * s.local({0: a.dag(), 1: e*zero.dag()}),
                            omega_s * sum([s.local({0: a, j: f*e.dag()}) for j in s.target_indices]
                           + [ratio * s.local({j: e*one.dag()}) for j in s.target_indices]),
                                -max_sideband * s.local({0: a.dag(), 1: e*zero.dag()})]

    
def worker_function(t):
    #print(t, end=" ")
    ratio = (t*omega_s/(np.pi) - 1)**-1
    
    execution_times = [np.pi/2/max_sideband, t - np.pi/max_sideband, np.pi/2/max_sideband]

    hamiltonians = H(omega_s, ratio)
    hamiltonians = [h+h.dag() for h in hamiltonians]
    
    # Compute fidelity
    return (get_process_fidelity(hamiltonians, execution_times, fanout, collapse_operators=[]))

fidelities = []

ratios = list(10**np.linspace(-1.7,-1,1000))



for k in np.arange(1,100):
    ratio1 = 1/2/k
    if ratio1 < np.max(ratios) and ratio1 > np.min(ratios):
        ratios.append(ratio1)
ratios.sort()

times = [np.pi/omega_s*(1/ratio1 + 1) for ratio1 in ratios]


# Compute fidelity with multiprocessing
bar = progressbar.ProgressBar(maxval=len(times), 
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' | ', progressbar.Timer(), ' | ', progressbar.ETA()])
fidelities = []
with multiprocessing.Pool(processes=16) as pool:    
    result = pool.imap(worker_function, times)
    
    # Update the progress bar as tasks complete
    bar.start()
    for i, fid in enumerate(result):
        bar.update(i+1)
        fidelities.append(fid)
    bar.finish()

plt.figure(figsize=(8,5))

label = "Timing Condition"
for k in np.arange(1,100):
    t = 2*np.pi*k/omega_s + np.pi/max_sideband
    if t < np.max(times) and t > np.min(times):
        color="black"
        plt.plot([1/2/k, 1/2/k], [np.min(1-np.array(fidelities))/2, np.max(1-np.array(fidelities))*2], color=color, linestyle="--", alpha=0.5, label=label)
        label = None  # turn off label for remaining lines
plt.xlabel("Ratio $\Omega_t / \Omega_c$ ")
plt.ylabel("Infidelity")

theoretical_infidelities = s.target_qubits * np.array(ratios) ** 2
timed_theoretical_infidelities = np.pi**2/16*(s.target_qubits)*(s.target_qubits-1)*(np.array(times)*omega_s/(np.pi) - 1)**-4
x= (np.array(times)*omega_s/(np.pi) - 1)**-1
k=5
temp = sum([comb(s.target_qubits, m)*4*m*x**2/(1+m*x**2)*np.sin((np.array(times) - np.pi/omega_s)/2 * omega_s*np.sqrt((1+m*x**2)))**2 for m in range(s.num_qubits)])/2**s.num_qubits

plt.loglog(ratios, temp, label="Theory",color="green", linestyle="-.", alpha=0.5)
plt.loglog(ratios, 1-np.array(fidelities), label="Simulation")
plt.plot(ratios, theoretical_infidelities, label="Upper Bound")
#plt.plot(ratios, timed_theoretical_infidelities, label="Timing Condition Fidelity")

plt.grid(axis="y")
plt.ylim(np.min(1-np.array(fidelities))/2, np.max(1-np.array(fidelities))*2)

plt.plot()
#omega_d = np.pi / np.array(times)
plt.legend()

plt.savefig("figures/fidelity-vs-ratio.pdf", bbox_inches='tight')
