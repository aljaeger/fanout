
from qutip import *
import numpy as np
from scipy.sparse import bsr_matrix

# Create a SeedSequence based on system entropy
seeds = np.random.SeedSequence()
options = Options(num_cpus=16)



def get_process_fidelity(hamiltonians, evolution_times, target_unitary, collapse_operators=None):
    """
    Calculate the fidelity of a quantum evolution.

    Parameters:
    - hamiltonians (list): A list of Hamiltonian operators representing the system's evolution.
    - evolution_times (list): A list of evolution times corresponding to each Hamiltonian.
    - collapse_operators (list): A list of collapse operators representing decoherence effects.

    Returns:
    float: The fidelity of the quantum evolution.
    """

    evolution_unitary = identity(target_unitary.dims[0])
    if collapse_operators == [] or collapse_operators is None:
        
        for time, hamiltonian in zip(evolution_times, hamiltonians):
            evolution_unitary *= (-1j * time * hamiltonian).expm()
        
        fidelity = abs((evolution_unitary*target_unitary.dag()).tr() / (target_unitary.dag()*target_unitary).tr())**2
    
    else:
        target_superoperator = sprepost(target_unitary, target_unitary)
        evolution_superoperator = identity((target_superoperator.dims)[0])

        
        for time, hamiltonian in zip(evolution_times, hamiltonians):
            Lio = liouvillian(hamiltonian, collapse_operators)
            evolution_superoperator *= (Lio * time).expm()
        
        evolution_superoperator = sprepost(target_unitary.dag()*target_unitary, target_unitary.dag()*target_unitary) * evolution_superoperator
        
        evolution_superoperator = to_choi(evolution_superoperator)
        kraus_ops = to_kraus(evolution_superoperator)
        
        tracecheck = sum([kraus_op.dag() * kraus_op for kraus_op in kraus_ops]) - target_unitary*target_unitary.dag()
                        
                        
        
        fidelity = sum([abs((target_unitary.dag()*kraus_op).tr())**2 for kraus_op in kraus_ops])/(target_unitary.dag()*target_unitary).tr()**2


    return np.abs(fidelity)


def OLDmontecarlo_process_fidelity(hamiltonians, evolution_times, target_unitary, collapse_operators, num_shots=100):
    
    fidelities = []
    for i in range(num_shots):
        pass
        random_state = rand_ket_haar(target_unitary.shape[0])
        random_state.dims= [target_unitary.dims[0], list(np.ones_like(target_unitary.dims[1]))]
        random_state = (target_unitary * random_state).unit() # state is random, applying a unitary wont change that, but projects on comp subspace
        target_state = target_unitary * random_state 
        
        for hamiltonian, time in zip(hamiltonians, evolution_times):
            random_state = (mcsolve(hamiltonian, random_state, np.linspace(0,time,100), [collapse_operators], ntraj=1, progress_bar = False).states[:,-1][0])
            #print(random_state)
        fidelities.append(expect(target_state*target_state.dag(), random_state))
    return np.mean(fidelities)

def montecarlo_process_fidelity(hamiltonians, evolution_times, target_unitary, collapse_operators, num_shots=100, initial_phonons = 0, a=None):
    
 
    # Generate a random seed from the SeedSequence
    seed = seeds.generate_state(1)[0]

    # Seed the NumPy random number generator
    np.random.seed(seed)

    fidelities = []
    
    for i in range(num_shots):
        random_state = rand_ket(target_unitary.shape[0])
        
        
        random_state.dims = [target_unitary.dims[0], list(np.ones_like(target_unitary.dims[1]))]
        random_state = (target_unitary * random_state).unit() # state is random, applying a unitary wont change that, but projects on comp subspace
        target_state = target_unitary * random_state 
        
        # heat random state to thermal occupation
        if a is not None:
            heating_op = np.sqrt(initial_phonons) * a.dag()
            cooling_op = np.sqrt(1 + initial_phonons) * a
            
            random_state = mesolve(hamiltonians[0]*0, random_state,np.linspace(0,10,20),c_ops=[heating_op, cooling_op]).states[-1]
        
        for hamiltonian, time in zip(hamiltonians, evolution_times):
            random_state = (mesolve(hamiltonian, random_state, np.linspace(0,time,100), c_ops=collapse_operators).states[-1])
            #print(random_state)
            
        
        fidelities.append(expect((target_state*target_state.dag()).ptrace(np.arange(1,len(random_state.dims[0][1:])+1)), random_state.ptrace(np.arange(1,len(random_state.dims[0][1:])+1))))
        #fidelities.append(expect((target_state*target_state.dag()), random_state))
    return np.mean(fidelities)



def basisstate_process_fidelity(hamiltonians, evolution_times, target_unitary, collapse_operators, basis_states):
    
    fidelities = []
    
    for initial_state in basis_states:
        initial_state = rand_ket(target_unitary.shape[0])
        initial_state.dims = [target_unitary.dims[0], list(np.ones_like(target_unitary.dims[1]))]
        initial_state = (target_unitary * initial_state).unit() # state is random, applying a unitary wont change that, but projects on comp subspace
        target_state = target_unitary * initial_state 

        for hamiltonian, time in zip(hamiltonians, evolution_times):
            initial_state = (mesolve(hamiltonian, initial_state, np.linspace(0,time,100), c_ops=collapse_operators).states[-1])
            #print(initial_state)
            
        
        fidelities.append(expect((target_state*target_state.dag()).ptrace(np.arange(1,len(initial_state.dims[0][1:])+1)), initial_state.ptrace(np.arange(1,len(initial_state.dims[0][1:])+1))))
    return np.mean(fidelities)

def get_average_gate_fidelity(hamiltonians, evolution_times, target_unitary, num_shots=10):
    
    for time, hamiltonian in zip(evolution_times, hamiltonians):
        pass
        

def sparsify(hamiltonian, computational_subspace):
    """Sets entries that do not couple to the computational subspace to 0"""
    
    dims = hamiltonian.dims
    index1, index2 = bsr_matrix(hamiltonian).nonzero()
        
    coupled_indices = set(bsr_matrix(computational_subspace).nonzero()[0])
    
    length = 0
    while True:
        relevant_indices = coupled_indices.intersection(index1)
        new_indices = []
        
        for index in relevant_indices:
            array_indices = np.where(index1 == index)
            coupled_indices = coupled_indices.union(index2[array_indices])
        
        if len(coupled_indices) == length:
            break
        else:
            length = len(coupled_indices)
    
    computational_subspace = np.zeros_like(hamiltonian)
    for index in coupled_indices:
        computational_subspace[index,index] = 1
    
    computational_subspace = Qobj(computational_subspace, dims=dims)
    
    return computational_subspace * hamiltonian * computational_subspace   
    
from itertools import combinations
from copy import deepcopy

def generate_variants(basis_digit, n_es, n_fs):
    one_indices = [i for i, bit in enumerate(basis_digit) if bit == 1]
    
    if n_es + n_fs > len(one_indices):
        raise ValueError("n_es + n_fs must be less than or equal to number of 1s in basis_digit.")
    
    results = []
    
    for e_indices in combinations(one_indices, n_es):
        remaining_ones = set(one_indices) - set(e_indices)
        for f_indices in combinations(remaining_ones, n_fs):
            temp = deepcopy(basis_digit)
            for i in e_indices:
                temp[i] = 2
            for i in f_indices:
                temp[i] = 3
            results.append(temp)
    
    return results

    
class StateSpace:
    
    def __init__(self, control_qubits, target_qubits,  num_phonons = 5, num_states = 4):
        self.control_qubits = control_qubits
        self.target_qubits = target_qubits
        self.num_phonons = num_phonons
        self.num_states = num_states
        
        num_qubits = control_qubits + target_qubits
        self.num_qubits = num_qubits
        
        # basic operations and local states
        self.internal_states = [fock(num_states, i) for i in range(num_states)]
        self.a = destroy(num_phonons)
        
        assert num_states >= 2, "must have at least 2 qubit states"
        if num_states == 2:
            self.zero, self.one = internal_states[:2]
        elif num_states == 3:
            self.zero, self.one, self.e = self.internal_states[:3]
        else:
            self.zero, self.one, self.e, self.f = self.internal_states[:4]
        
        self.minus, self.plus = (self.zero - self.one)/np.sqrt(2), (self.zero + self.one)/np.sqrt(2)
        
        # dimensions of the Hilbert space
        self.dims = [num_phonons] + num_qubits * [num_states]

        # indices
        self.motional_index = 0
        self.control_indices = np.arange(1, control_qubits + 1)
        self.target_indices = np.arange(control_qubits + 1, num_qubits + 1)

        # computational subspace
        self.basis_digits = np.array([[0] + [int(digit) for digit in bin(n)[2:].zfill(num_qubits)] for n in range(2**num_qubits)])
        self.basis_states = [basis(self.dims, list(digits)) for digits in self.basis_digits]
        self.basis_names = ["$|"+str(digit[1:])[1:-1]+"\\rangle$" for digit in self.basis_digits]
        
        # reduced basis to remove redundant states due to symmetry of the Hamiltonian
        self.basis_digits_reduced = [[0] + [0]*m2 + [1]*(control_qubits-m2) + [0]*m + [1]*(target_qubits-m) for m2 in range(control_qubits+1) for m in range(target_qubits+1) ]
        self.basis_states_reduced = [basis(self.dims, list(digits)) for digits in self.basis_digits_reduced]
        
        
    def local(self, operations_dict):
        """Defines a local operation on one or multiple qubits; returns the operation defined on the tensor product space

        operations_dict: dictionary containing index as keys, operations as values

        example: local({0: a, 1:e*zero.dag()}) is a red sideband on the first qubit (up to hermitian conjugate)"""
    
        total_operation = []
        for i in range(self.num_qubits+1):
            if i in operations_dict.keys():
                total_operation.append(operations_dict[i])
            else:
                total_operation.append(identity(self.dims[i]))

        return tensor(*total_operation)
    
    
    

s = StateSpace(1,7)
