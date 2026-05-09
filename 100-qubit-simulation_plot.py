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
from useful_functions import get_process_fidelity, sparsify, StateSpace, montecarlo_process_fidelity, basisstate_process_fidelity
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
# Matplotlib Plot Formatting
font = {'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)

# Set omega_t to match timing condition
gate_duration = np.pi

max_sideband = 20
omega_t = 1

print(f"data/qubitscalingsimulation_smallH{int(gate_duration*1000)}.p")
with open(f"data/qubitscalingsimulation_smallH{int(gate_duration*1000)}.p", "rb") as file:
    gate_duration, ns_extended, untimed_fidelities, n_fidelities = pickle.load(file)

print(len(ns_extended), len(untimed_fidelities))

plt.figure(figsize=(8,6))

# Calculate the mean and standard deviation
plot_ns = ns_extended[:len(untimed_fidelities)]

mean_fidelities = 1 - np.mean(untimed_fidelities, axis=1)
std_fidelities = np.std(untimed_fidelities, axis=1)


# Create gradient shading
y1 = mean_fidelities - 1 * std_fidelities
y2 = mean_fidelities + 1 * std_fidelities
#y1 = 1 - np.max(untimed_fidelities, axis=1)
#y2 = 1 - np.min(untimed_fidelities, axis=1)

"""# Outer lighter region (±2*std)
plt.fill_between(plot_ns, 1-(mean_fidelities + 1*std_fidelities), 
                1-(mean_fidelities - 1*std_fidelities), 
                 color="blue", alpha=0.1)
"""
plt.scatter(plot_ns, n_fidelities, color="green", label="Timed Fidelity")
# Inner darker region (±std)
plt.fill_between(plot_ns, 1-y1, 1-y2, color="blue", alpha=0.2, label="Untimed Fidelity")
plt.plot(plot_ns, np.array((plot_ns-1)*(omega_t / (max_sideband))**2), label="Theoretical Upper Bound", color="red")
plt.xticks(np.arange(int(plot_ns.min()), int(plot_ns.max())+1, 4))  # Adjust the step size as needed
plt.xlabel("Number of Qubits")
plt.ylabel("Gate Error")
plt.grid()
plt.xticks([2] + list(range(10, 101, 10)))

plt.legend()
plt.tight_layout()
plt.savefig("figures/100qubitsimulation.pdf")

plt.figure(figsize=(8,6))

# Calculate the mean and standard deviation
plot_ns = plot_ns[:30]
n_fidelities = n_fidelities[:30]
y1 = y1[:30]
y2 = y2[:30]

plt.scatter(plot_ns, n_fidelities, color="green", label="Timed Fidelity")
# Inner darker region (±std)
plt.fill_between(plot_ns, 1-y1, 1-y2, color="blue", alpha=0.2, label="Untimed Fidelity")
plt.plot(plot_ns, np.array((plot_ns-1)*(omega_t / (max_sideband))**2), label="Theoretical Upper Bound", color="red")
plt.xticks(np.arange(int(plot_ns.min()), int(plot_ns.max())+1, 4))  # Adjust the step size as needed
plt.xlabel("Number of Qubits")
plt.ylabel("Gate Error")
plt.grid()
plt.xticks(list(range(2, 31, 4)))

plt.legend()
plt.tight_layout()
plt.savefig("figures/30qubitsimulation.pdf")

