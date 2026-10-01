import torch, math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from collections import defaultdict
import numpy as np
from random import randint
import torch.nn.functional as F

def fs_distance(state1, state2):
    inner_product = torch.sum(state1 * state2.conj(), dim=1)
    return (torch.asin(inner_product.abs().clamp(0,1)))
    # return (torch.full(inner_product.size(), torch.pi/2) - torch.acos(inner_product.abs().clamp(0,1)))

def qcosine(bstates1, bstates2, eps=1e-9):
    norm1 = torch.linalg.vector_norm(bstates1, ord=2, dim=1)
    norm2 = torch.linalg.vector_norm(bstates2, ord=2, dim=1)
    inner_product = torch.sum(bstates1 * bstates2.conj(), dim=1)
    return inner_product.abs() / (norm1 * norm2 + eps)

def amplitude_encoding(vector):
    is_numpy = isinstance(vector, np.ndarray)
    is_1d = (vector.ndim == 1)

    if is_numpy: vector = torch.from_numpy(vector)
    vector = vector.to(torch.float64)
    device = vector.device
    if is_1d: vector = vector.unsqueeze(0)

    batch_size, dim = vector.shape

    num_qubits = math.ceil(math.log2(dim))
    target_dim = 1 << num_qubits

    if dim < target_dim:
        vector = F.pad(vector, (0, target_dim - dim), value=0.0)

    norm = torch.linalg.vector_norm(vector, ord=2, dim=-1, keepdim=True)

    is_zero = (norm == 0) 
    safe_norm = torch.where(is_zero, torch.ones_like(norm), norm)
    state_vector = vector / safe_norm

    state_vector[:, 0] = torch.where(is_zero.squeeze(-1), 
                                     torch.tensor(1.0, dtype=torch.float64, device=device), 
                                     state_vector[:, 0])
    
    final_norm = torch.linalg.vector_norm(state_vector, ord=2, dim=-1, keepdim=True)
    state_vector = state_vector / final_norm
    
    if is_1d: state_vector = state_vector.squeeze(0)
    if is_numpy: state_vector = state_vector.detach().cpu().numpy()

    return state_vector

def tn2qiskit(einsum_expr, gate_arr, param_dict={}, out_q=None, meas_output=True):
    if isinstance(einsum_expr, str):
        lhs, rhs = einsum_expr.split('->')
        input_indices = [list(sub.replace('$', '')) for sub in lhs.split(',')]
        output_indices = list(rhs.replace('$', ''))
    else:
        input_indices, output_indices = einsum_expr

    nq = sum(1 for gate in gate_arr if gate['op_type'] == '0')
    qreg = QuantumRegister(nq, f"qc{randint(1,10000)}")
    creg = ClassicalRegister(nq, f"c{randint(1,10000)}")
    qc = QuantumCircuit(qreg, creg)

    wire2q = defaultdict(list)
    qcounter = 0
    name2param = {}
    qiskit_param_dict = {}

    for idx_arr, gate in zip(input_indices, gate_arr):
        if gate['op_type'] == '0':
            wire_name = idx_arr[0]
            wire2q[wire_name].append(qcounter)
            qcounter += 1
        elif gate['op_type'] == '0_dag':
            # post_selection
            q_target = wire2q[idx_arr[0]].pop()
            qc.measure(q_target, q_target)
        else: 
            num_in = len(idx_arr) // 2 
            in_wires = idx_arr[:num_in]
            out_wires = idx_arr[num_in:]
            q_targets = [wire2q[w].pop(0) for w in in_wires]

            gate_func = getattr(qc, gate['op_type'].lower())
            if gate['name'] is None:
                gate_func(*q_targets)
            else:
                if gate['name'] in name2param:
                    param = name2param[gate['name']]
                else:
                    param = Parameter(gate['name'])
                    name2param[gate['name']] = param
                gate_func(param, *q_targets)

                if gate['name'] not in param_dict:
                    qiskit_param_dict[param] = np.random.rand() * 2 * np.pi
                else:
                    qiskit_param_dict[param] = param_dict[gate['name']]

            for w, q in zip(out_wires, q_targets):
                wire2q[w].append(q) 

    # bell-test
    for wire_name, q_list in wire2q.items():
        if len(q_list) == 2:
            q_left, q_right = q_list
            qc.cx(q_left, q_right)
            qc.h(q_left)
            qc.measure(q_left, q_left)
            qc.measure(q_right, q_right)

    output_qubits = []
    sentence_wires = output_indices[:out_q] if out_q is not None else output_indices

    for wire_name in sentence_wires:
        q_list = wire2q[wire_name]
        if len(q_list) == 1:
            q_out = q_list[0]
            if meas_output:
                qc.measure(q_out, q_out)
            output_qubits.append(q_out)
    
    return qc, output_qubits, qiskit_param_dict