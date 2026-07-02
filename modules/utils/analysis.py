from collections import defaultdict

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from opt_einsum import contract_path
import cotengra as ctg

from modules.utils.tensor_ops import einsum2interleaved, interleaved2einsum

def tn_metadata(data_arr):
    max_nq = max_gates = max_width = max_cdepth = 0
    avg_nq = avg_gates = avg_width = avg_cdepth = 0
    N = len(data_arr)
    path_cache = {}
    for einsum_expr, tarr in tqdm(data_arr):
        nq, ngates, cdepth, width = analyse_einsum(einsum_expr, tarr, cache=path_cache)
        max_nq = max(max_nq, nq)
        max_gates = max(max_gates, ngates)
        max_cdepth = max(max_cdepth, cdepth)
        max_width = max(max_width, width)
        avg_nq += nq
        avg_gates += ngates
        avg_cdepth += cdepth
        avg_width += width
    avg_width /= N
    avg_nq /= N
    avg_cdepth /= N
    avg_gates /= N
    return {'max': (max_nq, max_gates, max_cdepth, max_width ), 
            'avg': (int(round(avg_nq)), int(round(avg_gates)), int(round(avg_cdepth)), int(round(avg_width)))}

def analyse_einsum(einsum_expr, tarr, cache={}):
    op_types = tuple(op['op_type'] for op in tarr)
    cache_key = (einsum_expr, op_types)
    if cache_key in cache:
            return cache[cache_key]

    qubit_depths = defaultdict(int)
    shapes = []
    nq = 0

    input_subs, output_sub = einsum2interleaved(einsum_expr)
            
    for subscript, gate in zip(input_subs, tarr):
        symbol, op_type = gate['name'], gate['op_type']
        if symbol is None:
            if op_type == '0':
                data_shape = torch.Size([2])
                nq += 1
            elif op_type == '0_dag':
                data_shape = torch.Size([2])
            elif op_type == 'H': data_shape = torch.Size([2, 2])
            elif op_type == 'CX': data_shape = torch.Size([2, 2, 2, 2])
        else:
            if op_type in ['Rz', 'Rx', 'Ry']: data_shape = torch.Size([2, 2])
            elif op_type in ['CRz', 'CRx', 'CRy']: data_shape = torch.Size([2, 2, 2, 2])
        shapes.append(data_shape)

        if op_type not in ['0', 'sqrt']:
            current_gate_max = 0
            for char in subscript:
                current_gate_max = max(current_gate_max, qubit_depths[char])
            new_depth = current_gate_max + 1
            for char in subscript:
                qubit_depths[char] = new_depth

    cdepth = max(qubit_depths.values()) if qubit_depths else 0
    # opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy'], max_repeats=16, parallel=True)
    # tree = ctg.einsum_tree(einsum_str, *[tuple(int(d) for d in s) for s in shapes], optimize=opt)
    # max_width = tree.contraction_width()
    interleaved_args = []
    for shape, sub in zip(shapes, input_subs):
        interleaved_args.append(shape)
        interleaved_args.append(sub)
    interleaved_args.append(output_sub)
    path_info = contract_path(*interleaved_args, shapes=True)
    max_width = int(np.log2(float(path_info[1].largest_intermediate)))
    ngates = len(tarr) - nq
    cache[cache_key] = (nq, ngates, cdepth, max_width)
    return nq, ngates, cdepth, max_width