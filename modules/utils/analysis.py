from collections import defaultdict

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from opt_einsum import contract_path
import cotengra as ctg

from modules.utils.tensor_ops import einsum2interleaved, interleaved2einsum

def tn_metadata(data_arr):
    max_nq = max_gates = max_width = max_cdepth = max_2q_gates = 0
    avg_nq = avg_gates = avg_width = avg_cdepth = avg_2q_gates = 0
    N = len(data_arr)
    path_cache = {}
    i = 0
    for einsum_expr, tarr in tqdm(data_arr):
        i += 1
        # try:
        nq, ngates, cdepth, width, twoq_gates = analyse_einsum(einsum_expr, tarr, cache=path_cache)
        max_nq = max(max_nq, nq)
        max_gates = max(max_gates, ngates)
        max_cdepth = max(max_cdepth, cdepth)
        max_width = max(max_width, width)
        max_2q_gates = max(max_2q_gates, twoq_gates)
        avg_nq += nq
        avg_gates += ngates
        avg_cdepth += cdepth
        avg_width += width
        avg_2q_gates += twoq_gates
        # except Exception as e:
        #     print(f"Error analyzing einsum: {i} with error: {e}")
        #     continue
    avg_width /= N
    avg_nq /= N
    avg_cdepth /= N
    avg_gates /= N
    avg_2q_gates /= N
    return {'max': (max_nq, max_gates, max_cdepth, max_width, max_2q_gates), 
            'avg': (int(round(avg_nq)), int(round(avg_gates)), int(round(avg_cdepth)), int(round(avg_width)), int(round(avg_2q_gates)))}

def analyse_einsum(einsum_expr, tarr, cache={}):
    op_types = tuple(op['op_type'] for op in tarr)
    cache_key = (einsum_expr, op_types)
    if cache_key in cache:
            return cache[cache_key]

    qubit_depths = defaultdict(int)
    shapes = []
    nq = 0
    twoq_gates = 0

    input_subs = einsum_expr.split('->')[0]
            
    for subscript, gate in zip(input_subs, tarr):
        symbol, op_type = gate['name'], gate['op_type']
        if symbol is None:
            if op_type == '0':
                data_shape = torch.Size([2])
                nq += 1
            elif op_type == '0_dag':
                data_shape = torch.Size([2])
            elif op_type == 'H': data_shape = torch.Size([2, 2])
            elif op_type == 'CX': 
                data_shape = torch.Size([2, 2, 2, 2])
                twoq_gates += 1
        else:
            if op_type in ['Rz', 'Rx', 'Ry']: data_shape = torch.Size([2, 2])
            elif op_type in ['CRz', 'CRx', 'CRy']: 
                data_shape = torch.Size([2, 2, 2, 2])
                twoq_gates += 1
        shapes.append(data_shape)

        if op_type not in ['0']:
            current_gate_max = 0
            for char in subscript:
                current_gate_max = max(current_gate_max, qubit_depths[char])
            new_depth = current_gate_max + 1
            for char in subscript:
                qubit_depths[char] = new_depth

    cdepth = max(qubit_depths.values()) if qubit_depths else 0
    # dummy_operands = [np.empty(tuple(s), dtype=np.int8) for s in shapes]
    path_info = contract_path(einsum_expr, *shapes, shapes=True)
    max_width = int(np.log2(float(path_info[1].largest_intermediate)))
    ngates = len(tarr) - nq
    cache[cache_key] = (nq, ngates, cdepth, max_width, twoq_gates)
    return nq, ngates, cdepth, max_width, twoq_gates