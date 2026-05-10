import os

# Multithreading Options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from qutip import *
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from math import comb
import time
import pickle
import matplotlib
from scipy.sparse import csr_matrix

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
font = {'weight': 'normal', 'size': 18}
plt.rc('font', **font)

from useful_functions import get_process_fidelity, sparsify, StateSpace, montecarlo_process_fidelity, basisstate_process_fidelity


def make_sparse(quantum_object):
    quantum_object_dense = quantum_object.full()
    quantum_object_dims = quantum_object.dims
    quantum_object_sparse = csr_matrix(quantum_object_dense)
    return Qobj(quantum_object_sparse, dims=quantum_object_dims)


# Parameters
omega_t = 1
omega_s = 20
gate_duration = np.pi
num_phonons = 4
gamma = 0.01
n = 4
n_processes = 32

print(n, "qubit fanout simulation")

os.makedirs("figures", exist_ok=True)

# ── Full simulation ──────────────────────────────────────────────────────────

s = StateSpace(1, n-1, num_states=4, num_phonons=num_phonons+1)
a = s.a
zero, one, e, f = s.internal_states[:4]

H = lambda omega_s, omega_t: sum(
    [omega_s * s.local({0: a, j: f*e.dag()}) for j in s.target_indices] +
    [omega_t * s.local({j: e*one.dag()}) for j in s.target_indices]
)

hamiltonian = make_sparse(H(omega_s, omega_t) + H(omega_s, omega_t).dag())
c_ops = [make_sparse(s.local({0: a}) * np.sqrt(gamma)),
         make_sparse(s.local({0: a.dag()}) * np.sqrt(gamma))]

basis_transform = (
    s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons, 0).dag(), 1: e*zero.dag()})
    + s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: one*one.dag()}) / 2
)
basis_transform = basis_transform + basis_transform.dag()

label = "Full Simulation"
for basis_digit, initial_state in zip(s.basis_digits_reduced, s.basis_states_reduced):
    if basis_digit[1] == 0:
        initial_state = basis_transform * initial_state
    initial_state = make_sparse(initial_state)
    print(expect(s.local({0: a.dag()*a}), initial_state))
    final_states = mesolve(hamiltonian, initial_state, np.linspace(0, gate_duration, 200),
                           c_ops=c_ops, progress_bar=None).states
    fids = [(initial_state * initial_state.dag() * fs).tr() for fs in final_states]
    plt.plot(np.linspace(0, 1, 200), fids, alpha=0.5, color="black", label=label)
    label = None

# ── Block-diagonal simulation ────────────────────────────────────────────────

num_targets = n - 1
fidelities = [[], []]
label = "Block-Diagonal Simulation"

for control_state in [0, 1]:
    for num_ones in range(num_targets + 1):
        print("Now processing approximation for m=", num_ones, "out of", num_targets + 1)
        dims = [num_phonons+1, num_ones+1, num_ones+1]

        def Dicke_Hamiltonian():
            vec1 = basis(dims, [0, 0, 0])
            Hamiltonian = vec1 * vec1.dag() * 0
            for p in range(num_phonons+1):
                for num_es in range(num_ones+1):
                    for num_fs in range(num_ones+1):
                        state1 = [p, num_es, num_fs]
                        state2 = [p, num_es + 1, num_fs]
                        if not all(0 <= state1[i] < dims[i] for i in range(3)) or \
                           not all(0 <= state2[i] < dims[i] for i in range(3)):
                            continue
                        multiplier = np.sqrt((num_es+1) * (num_ones - num_es - num_fs))
                        if num_ones - num_es - num_fs <= 0:
                            continue
                        v1 = basis(dims, state1)
                        v2 = basis(dims, state2)
                        Hamiltonian += omega_t * multiplier * (v2*v1.dag() + v1*v2.dag())

            for p in np.arange(0, num_phonons+1):
                for num_es in np.arange(0, num_ones+1):
                    for num_fs in range(0, num_ones+1):
                        state1 = [p, num_es, num_fs]
                        state2 = [p-1, num_es - 1, num_fs + 1]
                        if not all(0 <= state1[i] < dims[i] for i in range(3)) or \
                           not all(0 <= state2[i] < dims[i] for i in range(3)):
                            continue
                        multiplier = np.sqrt((num_fs+1) * (num_es - num_fs) * p)
                        if num_es - num_fs <= 0:
                            continue
                        v1 = basis(dims, state1)
                        v2 = basis(dims, state2)
                        Hamiltonian += omega_s * multiplier * (v2*v1.dag() + v1*v2.dag())
            return Hamiltonian

        def creation_op():
            vec1 = basis(dims, [0, 0, 0])
            c_op = vec1 * vec1.dag() * 0
            for p in np.arange(0, num_phonons+1):
                for num_es in np.arange(0, num_ones+1):
                    for num_fs in range(0, num_ones+1):
                        state1 = [p, num_es, num_fs]
                        state2 = [p+1, num_es, num_fs]
                        if not all(0 <= state1[i] < dims[i] for i in range(3)) or \
                           not all(0 <= state2[i] < dims[i] for i in range(3)):
                            continue
                        multiplier = np.sqrt(p+1)
                        c_op += multiplier * basis(dims, state2) * basis(dims, state1).dag()
            return c_op

        h_block = Dicke_Hamiltonian()
        c_ops_block = [creation_op() * np.sqrt(gamma), creation_op().dag() * np.sqrt(gamma)]
        basis_vec = basis(dims, [1 - control_state, 0, 0])
        states = mesolve(h_block, basis_vec, np.linspace(0, gate_duration, 200),
                         c_ops=c_ops_block).states
        fids = [(basis_vec * basis_vec.dag() * state).tr() for state in states]

        if control_state == 0:
            fidelities[0].append(np.array(fids))
        else:
            fidelities[1].append(np.array(fids))

        plt.plot(np.linspace(0, 1, 200), fids, linestyle=":", color="blue", label=label)
        label = None

plt.legend()
plt.xlabel("Gate Evolution")
plt.ylabel("Fidelity")
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, left=0.12)
plt.savefig(f"figures/test_{n}.pdf")
plt.close()

# ── Multiprocessed scatterplot evaluation ───────────────────────────────────

def evaluate_single_state(args):
    i, amplitudes_i, phases_i, fidelities, omega_s, omega_t, gate_duration, gamma, num_phonons, n = args

    from scipy.sparse import csr_matrix
    from qutip import Qobj, fock, mesolve, expect
    from useful_functions import StateSpace

    def make_sparse_local(qobj):
        return Qobj(csr_matrix(qobj.full()), dims=qobj.dims)

    s = StateSpace(1, n-1, num_states=4, num_phonons=num_phonons+1)
    a = s.a
    zero, one, e, f = s.internal_states[:4]

    H = lambda os, ot: sum(
        [os * s.local({0: a, j: f*e.dag()}) for j in s.target_indices] +
        [ot * s.local({j: e*one.dag()}) for j in s.target_indices]
    )
    hamiltonian = make_sparse_local(H(omega_s, omega_t) + H(omega_s, omega_t).dag())
    c_ops = [make_sparse_local(s.local({0: a}) * np.sqrt(gamma)),
             make_sparse_local(s.local({0: a.dag()}) * np.sqrt(gamma))]

    basis_transform = (
        s.local({0: fock(s.num_phonons, 1)*fock(s.num_phonons, 0).dag(), 1: e*zero.dag()})
        + s.local({0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: one*one.dag()}) / 2
    )
    basis_transform = make_sparse_local(basis_transform + basis_transform.dag())

    fanout_transition = s.local(
        {0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.one*s.one.dag()} |
        {j: s.zero*s.zero.dag() - s.one*s.one.dag() for j in s.target_indices}
    )
    fanout_idle = s.local(
        {0: fock(s.num_phonons, 0)*fock(s.num_phonons, 0).dag(), 1: s.zero*s.zero.dag()} |
        {j: s.zero*s.zero.dag() + s.one*s.one.dag() for j in s.target_indices}
    )
    fanout = make_sparse_local(fanout_idle + fanout_transition)

    initial_state = sum([amp * phase * state
                         for amp, phase, state in zip(amplitudes_i, phases_i, s.basis_states)])
    initial_state = make_sparse_local(initial_state)

    target_state = make_sparse_local(basis_transform * fanout * initial_state)
    initial_state = make_sparse_local(basis_transform * initial_state)

    final_state = mesolve(hamiltonian, initial_state, np.linspace(0, gate_duration, 200),
                          c_ops=c_ops, progress_bar=None).states[-1]
    exact_fid = float(np.abs((target_state * target_state.dag() * final_state).tr()))

    num_ones_array = [int(np.sum(digits[2:])) for digits in s.basis_digits]
    control_states_array = [digits[1] for digits in s.basis_digits]
    approx_fid = float(np.abs(
        sum([fidelities[control_states_array[j]][num_ones_array[j]] * amplitudes_i[j]**2
             for j in range(len(amplitudes_i))])[-1]
    ))

    return exact_fid, approx_fid


num_trials = 100

all_amplitudes = []
all_phases = []
for _ in range(num_trials):
    amplitudes = np.random.normal(0, 1, size=2**n)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    phases = np.exp(1j * np.random.uniform(0, 2*np.pi, size=2**n))
    all_amplitudes.append(amplitudes)
    all_phases.append(phases)

args_list = [
    (i, all_amplitudes[i], all_phases[i], fidelities, omega_s, omega_t,
     gate_duration, gamma, num_phonons, n)
    for i in range(num_trials)
]

with multiprocessing.Pool(processes=n_processes) as pool:
    results = list(tqdm(pool.imap(evaluate_single_state, args_list), total=num_trials))

exact_fids, approx_fids = zip(*results)
exact_fids = list(exact_fids)
approx_fids = list(approx_fids)

# ── Final scatterplot ────────────────────────────────────────────────────────

plt.grid()
lims = [min(np.min(approx_fids), np.min(exact_fids)),
        max(np.max(approx_fids), np.max(exact_fids))]
plt.plot(lims, lims, color="blue", alpha=0.5, linestyle="--", label="Ideal Approximation")
plt.scatter(exact_fids, approx_fids, color="red", label="Random State Simulation")
plt.xlabel("Exact Fidelity")
plt.ylabel("Basis State Approximation")
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/test_heating_scatterplot_{n}.pdf")
plt.close()

rmse = np.sqrt(np.mean((np.array(approx_fids) - np.array(exact_fids))**2))
print("RMSE:", rmse)