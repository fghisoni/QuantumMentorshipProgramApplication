import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def SWAP(wire1, wire2, p):
    ''' 
    Function to implement the SWAP of 2 wires as a combination of 3 CNOTs. 
    Each CNOT induces a polarization channel on both qubits

    wire1 (array[complex]): First wire to be swapped
    wire2 (array[complex]): Second wire to be swapped
    p (float): qubit gate error
    '''

    qml.CNOT([wire1, wire2])
    depolarization([wire1, wire2], p)
    qml.CNOT([wire2, wire1])
    depolarization([wire1, wire2], p)
    qml.CNOT([wire1, wire2])
    depolarization([wire1, wire2], p)

def fidelity(density1, density2):
    ''' 
    Calcuates the fidelity between 2 density matrices.

    density1 (array[complex]): Density matrix of a quantum state
    density2 (array[complex]): Density matrix of a quantum state
    '''

    return qml.math.fidelity(density1, density2)

def depolarization(wires, p):
    ''' 
    wires (list(int)): list of wires to apply a depolarization channel to
    p (float): qubit gate error
    '''
    for i in wires:
        qml.DepolarizingChannel(p, wires=i)


def SWAP_circuit(p, n_qubits, list_ops_state_1, list_ops_state_2):
    ''' 
    Creates the SWAP circuit between 2 quantum states.
    Each gate introduces a polarization channel on the qubit.

    p (float): qubit gate error
    n_qubits (int): number of qubits
    list_ops_state_1 (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for state 1 
    list_ops_state_2 (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for state 2 
    '''
    for op1, op2 in zip(list_ops_state_1, list_ops_state_2):
        qml.apply(op1)
        # print(op1.wires[:])
        depolarization(op1.wires[:], p)
        qml.apply(op2)
        depolarization(op2.wires[:], p)

    for j in range(n_qubits//2):
        SWAP(j,j + n_qubits//2, p)
        
    return qml.state()

def create_operations_list(states, n_qubits_list):
    ''' 
    Creates a list with the operations required to perform Mottonen State Preparation for the states given as input.

    states (list[array(complex)]): The states to be prepared by the Mottonen State Preparation
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    operations_list = [
                    [
                        qml.MottonenStatePreparation(states[i][0], wires=range(n_qubits_list[i]//2)).decomposition(),
                        qml.MottonenStatePreparation(states[i][1], wires=range(n_qubits_list[i]//2, n_qubits_list[i])).decomposition()
                        ]
                   for i in range(len(states))
                   ]
    return operations_list

def create_device_list(n_qubits_list, shots):
    ''' 
    Creates all the devices with the required number of qubits for the experiments

    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    if shots == 'None':
        dev_list = [qml.device('default.mixed', wires=n_qubits_list[i]) for i in range(len(n_qubits_list))]
    else:
        dev_list = [qml.device('default.mixed', wires=n_qubits_list[i], shots=shots) for i in range(len(n_qubits_list))]
    return dev_list

def create_circuit_list(devices, n_qubits_list):
    ''' 
    Create the list of QNodes with a SWAP circuit and the accoding device
    
    devices (list[qml.device]): List of devices to attach the QNodes to.
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    circuits = [qml.QNode(SWAP_circuit, devices[i]) for i in range(len(n_qubits_list))] 
    return circuits

def convert_to_statevector(state1, state2):
    ''' 
    Given the state vectors state1 and state2 the function will return the statevector of the quantum state with both states.

    state1 (array[complex]): State vector of a quantum system
    state2 (array[complex]): State vector of a quantum system
    '''
    state_vec = [i*j for i in state1 for j in state2]
    return np.array(state_vec)

def extract_statvector(density):
    ''' 
    Given a density matrix the function will return the according state vector

    density (array[complex]): Density matrix of a quantum state
    '''
    state_vec = [[np.sqrt(density[j][i,i]) for i in range(density[j].shape[0])] for j in range(len(density))]
    return np.array(state_vec)

def density_matrix_results(circuits, n_qubits_list, operations_list, p_values):
    ''' 
    Run the circuits which will return the density matrix. The function will be run for all circuit sizes alongside all combinations of p values.

    circuits (list[qml.QNode]): List of all the circuits required for the experiment
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    operations_list (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for all states
    p_values (np.array): List of gate error probabilities to test 
    '''
    res = [
        [circuits[i](
            p = j, 
            n_qubits = n_qubits_list[i], 
            list_ops_state_1 = operations_list[i][0], 
            list_ops_state_2 = operations_list[i][1]
            ) 
        for i in range(len(n_qubits_list))
        ] 
    for j in p_values
    ]
    return np.array(res)

def calc_fidelity(zeros_noise_results, noisy_results, n_qubits_list,  p_values):
    ''' 
    Calculates the fidelity between the density matrices of the noiseless SWAP circuits and all the noisy SWAP circuits.

    zeros_noise_results (list[array]): Density matrices for all the circuits with no noise
    noisy_results (list[array]): Density matrices for all the noisy simulations 
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    p_values (np.array): List of gate error probabilities to test 
    '''
    fidelties = [[fidelity(noisy_results[i][j], zeros_noise_results[0][j]) for j in range(len(n_qubits_list))] for i in range(len(p_values))]
    return np.array(fidelties)

def create_specs_list(circuits, n_qubits_list, operations_list):
    ''' 
    Create a list containing the spec of all the circuits used in the simulation

    circuits (list[qml.QNode]): List of all the circuits required for the experiment
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    operations_list (list(qml.operations)): List of Pennylane Operations to implement the Mottonen State Preparation for all states
    '''
    specs_fun = [qml.specs(circuits[i])(0.5, n_qubits_list[0], operations_list[i][0], operations_list[i][1]) for i in range(len(n_qubits_list))]
    return specs_fun

def create_num_gate_list(specs_fun, n_qubits_list):
    ''' 
    Retruns a list containg the number of gates in each circuit

    specs_fun (list[qml.specs]): List of the specs of all the circuits used in the experiment
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    num_gates = [specs_fun[i]['resources'].num_gates - specs_fun[i]['resources'].gate_types['DepolarizingChannel'] for i in range(len(n_qubits_list))]
    return num_gates

def create_num_depolarization_gates_list(specs_fun, n_qubits_list):
    ''' 
    Retruns a list containg the number of polarization channels in each circuit

    specs_fun (list[qml.specs]): List of the specs of all the circuits used in the experiment
    n_qubits_list (list[int]): List of with the number of qubits for each investigation
    '''
    num_polarization_gates = [specs_fun[i]['resources'].gate_types['DepolarizingChannel'] for i in range(len(n_qubits_list))]
    return num_polarization_gates
