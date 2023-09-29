import pennylane as qml
from pennylane import numpy as np

from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

def create_random_quantum_state(n_qubits_list):
    ''' 
    Create a list of normalized quantum states with real and complex parts.

    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    quantum_states = []
    for i in n_qubits_list:
        state1 = np.random.rand(2**(i//2))
        state1 = state1/np.linalg.norm(state1)

        state2 = np.random.rand(2**(i//2))
        state2 = state2/np.linalg.norm(state2)
        
        quantum_states.append([state1, state2])
    return quantum_states

def SWAP_IBM(wire1, wire2):
    ''' 
    Function to implement the SWAP of 2 wires as a combination of 3 CNOTs. 

    wire1 (array[complex]): First wire to be swapped
    wire2 (array[complex]): Second wire to be swapped
    '''

    qml.CNOT([wire1, wire2])
    qml.CNOT([wire2, wire1])
    qml.CNOT([wire1, wire2])

def fidelity(density1, density2):
    ''' 
    Calcuates the fidelity between 2 density matrices.

    density1 (array[complex]): Density matrix of a quantum state
    density2 (array[complex]): Density matrix of a quantum state
    '''

    return qml.math.fidelity(density1, density2)


def SWAP_circuit_IBM(n_qubits, list_ops_state_1, list_ops_state_2):
    ''' 
    Creates the SWAP circuit between 2 quantum states.

    n_qubits (int): number of qubits
    list_ops_state_1 (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for state 1 
    list_ops_state_2 (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for state 2 
    '''
    for op1, op2 in zip(list_ops_state_1, list_ops_state_2):
        qml.apply(op1)
        qml.apply(op2)
    for j in range(n_qubits//2):
        SWAP_IBM(j,j + n_qubits//2)
        
    return qml.probs()

def create_device_list_IBM(n_qubits_list, provider, shots, fake_backend):
    ''' 
    Creates all the devices with the required number of qubits for the experiments

    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''

    backend = provider.get_backend('ibmq_qasm_simulator')
    backend._options.update_options( noise_model = NoiseModel.from_backend(fake_backend), seed_simulator=42)
    
    dev_list = [qml.device('qiskit.ibmq.sampler', wires=n_qubits_list[i], backend=backend, provider=provider,
                    shots=shots,seed_simulator=42,seed_transpiler=42,) for i in range(len(n_qubits_list))]
    return dev_list


def create_circuit_list_IBM(devices, n_qubits_list):
    ''' 
    Create the list of QNodes with a SWAP circuit and the accoding device
    
    devices (list[qml.device]): List of devices to attach the QNodes to.
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    circuits = [qml.QNode(SWAP_circuit_IBM, devices[i]) for i in range(len(n_qubits_list))] 
    return circuits

def probs_results(circuits, n_qubits_list, operations_list):
    ''' 
    Run the circuits which will return the probability distribution. The function will be run for all circuit sizes.

    circuits (list[qml.QNode]): List of all the circuits required for the experiment
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    operations_list (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for all states
    '''
    res = [circuits[i](
            n_qubits = n_qubits_list[i], 
            list_ops_state_1 = operations_list[i][0], 
            list_ops_state_2 = operations_list[i][1]
            ) 
        for i in range(len(n_qubits_list))
        ] 
    return np.array(res)

def calc_fidelity_IBM(zeros_noise_results, results, n_qubits_list):
    ''' 
    Calculates the fidelity between the density matrices of the noiseless SWAP circuits and all the noisy SWAP circuits.

    zeros_noise_results (list[array]): Density matrices for all the circuits with no noise
    noisy_results (list[array]): Density matrices for all the noisy simulations 
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    
    results_state = [np.sqrt(i) for i in results]
    density_matrix_results = [np.outer(results_state[i], results_state[i]) for i in range(len(n_qubits_list))]
    return [fidelity(zeros_noise_results[0][i], density_matrix_results[i]) for i in range(len(n_qubits_list))]