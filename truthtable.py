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

s = StateSpace(1,3, num_states=4, num_phonons=4)
a = s.a
zero, one, e, f = s.internal_states[:4]

fanout_transition = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.one*s.one.dag()} | {i: s.zero*s.zero.dag() - s.one*s.one.dag() for i in s.target_indices})
fanout_idle = s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.zero*s.zero.dag()} | {i: s.zero* s.zero.dag() + s.one*s.one.dag() for i in s.target_indices})
fanout = fanout_idle + fanout_transition

basis_transform = s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons,0).dag(), 1: e*zero.dag()}) + s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons,0).dag(), 1: one*one.dag()})

gate_duration=0.1
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



populations = np.zeros((len(s.basis_states), len(s.basis_states)+1))
phases = np.zeros((len(s.basis_states), len(s.basis_states)+1))

evolution = (hamiltonian * -1j * (gate_duration - np.pi/max_sideband)).expm()
print(gate_duration)
print()

for i, basis_state in enumerate(s.basis_states):
    basis_state = basis_transform * basis_state
    basis_state = evolution * basis_state #mesolve(hamiltonian, basis_state, np.linspace(0, gate_duration-np.pi/max_sideband,200), progress_bar=None).states[-1]
    basis_state = basis_transform.dag() * basis_state
    
    for j, basis_state2 in enumerate(s.basis_states):
        overlap = (basis_state.dag()*basis_state2)
        populations[i,j] = np.abs(overlap)**2
        phases[i,j] = np.arctan2(overlap.imag, overlap.real)
print(populations)


for i in range(len(s.basis_states)):
    populations[i,-1] = 1 - np.sum(populations[i,:-1])

populations[populations==0] = 1e-10

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable

# Sample data (you should replace this with your actual data)
# Ensure 's.basis_states', 'populations', and 'phases' are defined somewhere in your code

# Initialize the target_populations and log transformations
target_populations = np.zeros((len(s.basis_states), len(s.basis_states) + 1))
logtarget_populations = np.zeros_like(target_populations)
logpopulations = np.zeros_like(target_populations)
cutoff = 1e-6  # Some cutoff value for log scaling

for i in range(len(s.basis_states)):
    target_populations[i, i] = 1

for i in range(len(s.basis_states)):
    for j in range(len(s.basis_states) + 1):
        logtarget_populations[i, j] = np.log10(max(target_populations[i, j], cutoff)) - np.log10(cutoff)
        logpopulations[i, j] = np.log10(max(populations[i, j], cutoff)) - np.log10(cutoff)

# Create figure
fig = plt.figure(figsize=(18, 15))
ax = fig.add_subplot(111, projection='3d')

# Prepare 3D grid for the plot
xpos, ypos = np.meshgrid(np.arange(len(s.basis_states) + 1), np.arange(len(s.basis_states)))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = dy = 0.6 * np.ones_like(zpos)
dz = populations.flatten()
phases = phases.flatten()

# Define colormap based on phases
colors = plt.cm.hsv(phases / (2 * np.pi))  # Convert phases to colors using hsv colormap

# Plot bars with colors based on phases (non-zero dz values only)
ax.bar3d(xpos[dz > 1e-6], ypos[dz > 1e-6], zpos[dz > 1e-6], 
          dx[dz > 1e-6], dy[dz > 1e-6], dz[dz > 1e-6], 
          color=colors[dz > 1e-6], zsort='average')

# Colorbar (phase-based)
sm = ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(0, 2 * np.pi))
sm.set_array([])


# Create colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6, ticks=[0, np.pi, 2*np.pi])
cbar.set_ticklabels(["$0$", "$\pi$", "$2\pi$"])
cbar.set_label('Phase')

# Adjust colorbar position: move it closer to the plot
cbar.ax.set_position([0.1, 0.1, 1, 0.02])  # Adjust values to move it closer or further

# Plot the target populations (as bars)
dz = target_populations.flatten()
colors = plt.cm.hsv(phases / (2 * np.pi))  # Reuse phases for color

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', alpha=0.2)

# Labeling axes with padding adjustments
ax.set_xlabel('Output State', labelpad=45)
ax.set_ylabel('Input State', labelpad=35)
ax.set_zlabel('Population', labelpad=15)

# Customizing ticks and tick labels for better readability
ax.set_yticks(np.arange(len(s.basis_states)))
ax.set_yticklabels(s.basis_names, rotation=-20, ha="left", va="center")
ax.set_xticks(np.arange(len(s.basis_states) + 1))
ax.set_xticklabels(s.basis_names + ["Leakage"], rotation=45, ha="right", va="top")

# Optional: Print the sum of diagonal populations (for diagnostics)
print(np.sum([populations[i, i] for i in range(len(s.basis_states))]) / len(s.basis_states))

# Adjust the layout for better spacing
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)

plt.savefig("figures/fanout-truthtable.pdf", bbox_inches='tight')