import pandas as pd
from itertools import count
from collections import Counter
from abc import ABC, abstractmethod
from tqdm import tqdm
from modules.utils.tensor_ops import interleaved2einsum, sort_tn
from modules.compilation.quantum.gates import *

class BaseAnsatz(ABC):
    def __init__(self, obmap=set(), layers=1):
        super().__init__()
        self.obmap = obmap
        self.layers = layers
        self.id = type(self).__name__ + '_' + str(obmap['n']) + '_' + str(obmap['s']) + '_' + str(obmap['p']) + '_' + str(obmap['out'])
        self.char_idx = count(0)

    def __call__(self, tn, curry=False, discard=False, spider=False):
        if curry:
            return self.tn2ansatz_curried(tn,  discard=discard)
        elif spider:
            return self.spider(tn)
        else:
            return self.tn2ansatz(tn, discard=discard)

    def tns2ansatze(self, tn_arr, curry=False, discard=False, spider=False):
        return [self(tn, curry=curry, discard=discard, spider=spider) for tn in tn_arr]

    def reset_char(self):
        self.char_idx = count(0)

    def get_char(self):
        i = next(self.char_idx)
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return chars[i] if i < len(chars) else chr(192 + i - len(chars))

    def ccg_map(self, tn):
        ccg_map = {}
        for _, idx_arr, type_arr in tn:
            for idx, typ in zip(idx_arr, type_arr):
                if idx not in ccg_map:
                    ccg_map[idx] = [self.get_char() for _ in range(self.obmap.get(typ, 1))]
        return ccg_map

    @abstractmethod
    def ansatz(self, current_wires, base_symbol):
        pass

    def gen_einsum_expr(self, input_indices, ccg_map):
        all_idx = [i for sub in input_indices for i in sub]
        idx_counts = Counter(all_idx)
        output_indices = [idx for idx, cnt in idx_counts.items() if cnt == 1]

        if 0 in ccg_map:
            root_indices = ccg_map[0]
        elif '0' in ccg_map:
            root_indices = ccg_map['0']
        elif len(ccg_map) > 0:
            root_indices = list(ccg_map.values())[0]
        else:
            root_indices = []

        # root_indices = ccg_map.get(0, ccg_map.get('0', []))
        out_list = [i for i in root_indices if i in output_indices] + \
                   [i for i in output_indices if i not in root_indices]
        
        return interleaved2einsum(input_indices, out_list)
    
    def tn2ansatz(self, tn, discard=False):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []

        for word, idx_arr, type_arr in tn:
            N = sum(self.obmap.get(typ, 1) for typ in type_arr)
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append([w])
                # tensor_arr.append((None, '0'))
                tensor_arr.append({'name': None, 'op_type': '0'})

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                if idx not in ccg_map:
                    ccg_map[idx] = current_wires[i:i+n]
                else:
                    if not discard:
                        target_wires = ccg_map[idx]                    
                        replace_map = dict(zip(current_wires[i:i+n], target_wires))
                        input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
                i += n
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr
    
    def tn2ansatz_curried(self, tn, discard=False):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []
        tn_sorted = sort_tn(tn)

        for word, idx_arr, type_arr in tn_sorted:
            current_wires = []

            input_wires, input_types = [], []
            output_wires, output_types = [], []
            for idx, atomic_type in zip(idx_arr, type_arr):
                if idx in ccg_map:
                    input_wires.append(idx)
                    input_types.append(atomic_type)
                else:
                    output_wires.append(idx)
                    output_types.append(atomic_type)
            if len(input_wires) == len(output_wires):
                for in_idx, out_idx in zip(input_wires, output_wires):
                    current_wires.extend(ccg_map[in_idx])
                    ccg_map[out_idx] = ccg_map[in_idx]
                    del ccg_map[in_idx]
            else:
                for idx, atomic_type in zip(input_wires, input_types):
                    current_wires.extend(ccg_map[idx])
                for idx, atomic_type in zip(output_wires, output_types):
                    n = self.obmap.get(atomic_type, 1)
                    new_wires = [self.get_char() for _ in range(n)]
                    current_wires.extend(new_wires)
                    ccg_map[idx] = new_wires
                    input_indices.extend([[w] for w in new_wires])
                    tensor_arr.extend([{'name': None, 'op_type': '0'}] * n)

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend([[idx for idx in ten] for ten in new_indices])
            tensor_arr.extend(new_tensors)
            
            i = 0
            for idx, typ in zip(idx_arr, type_arr):
                n = self.obmap.get(typ, 1)
                ccg_map[idx] = current_wires[i:i+n]
                i += n

        if not discard:
            for virtual_idx, physical_wires in ccg_map.items():
                if virtual_idx != 0:
                    input_indices.extend([[w] for w in physical_wires])
                    # tensor_arr.extend([(None, '0_dag')] * len(physical_wires))
                    tensor_arr.extend([{'name': None, 'op_type': '0_dag'}] * len(physical_wires))
        
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        return einsum_expr, tensor_arr
    
    def spider(self, tn):
        self.reset_char()
        ccg_map = {}
        input_indices, tensor_arr = [], []

        idx = 0
        N = self.obmap.get('out', 1)
        for word, _, type_arr in tn:
            current_wires = [self.get_char() for _ in range(N)]

            for w in current_wires:
                input_indices.append([w])
                tensor_arr.append({'name': None, 'op_type': '0'})

            base_symbol = f"{word}__{'@'.join(type_arr)}"
            current_wires, new_indices, new_tensors = self.ansatz(current_wires, base_symbol)
            input_indices.extend(new_indices)
            tensor_arr.extend(new_tensors)

            ccg_map[idx] = current_wires
            idx += 1

        target_wires = ccg_map[0]                    
        replace_map = {}
        for i in range(idx):
            for j, output_wires in enumerate(ccg_map[i]):
                replace_map[output_wires] = target_wires[j]
        input_indices = [[replace_map.get(w, w) for w in sub] for sub in input_indices]
        for i in range(idx):
            ccg_map[i] = target_wires
        einsum_expr = self.gen_einsum_expr(input_indices, ccg_map)
        lhs, _ = einsum_expr.split('->')
        einsum_expr = f"{lhs}->{''.join(target_wires)}"
        return einsum_expr, tensor_arr

    def compile_dataset(self, df, curry=False, discard=False, spider=False):
        #blueprint_df = pd.DataFrame(index=df.index)
        cols = [col for col in df.columns if col.endswith('_diagram')]
        for col in cols:
            base_label = col.replace('_diagram', '')
            einsum_col = f"{base_label}_einsum"
            symbols_col = f"{base_label}_symbols"

            first_val = df[col].iloc[0]
            is_nested_list = (isinstance(first_val, list) 
                              and len(first_val) > 0 
                              and isinstance(first_val[0], list)
                             )
            einsum_arr, symbols_arr = [], []

            for row_val in tqdm(df[col]):
                if is_nested_list:
                    row_einsums, row_symbols = [], []
                    for tn in row_val:
                        e_str, syms = self(tn, curry, spider)
                        row_einsums.append(e_str)
                        row_symbols.append(syms)
                    einsum_arr.append(row_einsums)
                    symbols_arr.append(row_symbols)
                else:
                    e_str, syms = self(row_val, curry, spider)
                    einsum_arr.append(e_str)
                    symbols_arr.append(syms)
            df[einsum_col] = einsum_arr
            df[symbols_col] = symbols_arr
        return df
                        
class CustomV6Ansatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for i in range(N):
            nxt = self.get_char()
            new_indices.append(current_wires[i] + nxt)
            new_tensors.append({'name': None, 'op_type': 'H'})
            current_wires[i] = nxt

        for l in range(self.layers):
            op_idx = 0
            for i in range(N):
                gate_seq = ['Rx', 'Ry', 'Rx'] if (i == N - 1 and N > 1) else ['Rz', 'Ry', 'Rz']
                for g in gate_seq:
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': g})
                    current_wires[i] = nxt
                    op_idx += 1
                    
            if N > 1:
                for i in range(N):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)

                    cr_type = 'CRx' if (i == N - 1) else 'CRz'
                    new_tensors.append({'name': f"{base_symbol}_ent_l{l}_{op_idx}", 'op_type': cr_type})
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1

        return current_wires, new_indices, new_tensors

class CustomV5Ansatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for i in range(N):
            nxt = self.get_char()
            new_indices.append(current_wires[i] + nxt)
            # new_tensors.append((None, 'H'))
            new_tensors.append({'name': None, 'op_type': 'H'})
            current_wires[i] = nxt

        for l in range(self.layers):
            op_idx = 0
            for i in range(N):
                for g in ['Rz', 'Ry', 'Rz']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", g))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': g})
                    current_wires[i] = nxt
                    op_idx += 1
            if N > 1:
                for i in range(N):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    # new_tensors.append((None, 'CX'))
                    new_tensors.append({'name': None, 'op_type': 'CX'})
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
        return current_wires, new_indices, new_tensors
    
class IQPAnsatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for l in range(self.layers):
            op_idx = 0
            for i in range(N):
                nxt = self.get_char()
                new_indices.append(current_wires[i] + nxt)
                # new_tensors.append((None, 'H'))
                new_tensors.append({'name': None, 'op_type': 'H'})
                current_wires[i] = nxt

                nxt = self.get_char()
                new_indices.append(current_wires[i] + nxt)
                # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Rz'))
                new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'Rz'})
                current_wires[i] = nxt
                op_idx += 1

            if N > 1:
                for i in range(N):
                    c_idx, t_idx = i, (i + 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRz'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'CRz'})
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1

        for i in range(N):
            nxt = self.get_char()
            new_indices.append(current_wires[i] + nxt)
            # new_tensors.append((None, 'H'))
            new_tensors.append({'name': None, 'op_type': 'H'})
            current_wires[i] = nxt
            
        return current_wires, new_indices, new_tensors
        
class BrickworkAnsatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for l in range(self.layers):
            op_idx = 0
            if N > 1:
                for i in range(N):
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'Ry'})                    
                    current_wires[i] = nxt
                    op_idx += 1

                for i in range(N-1):
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[i] + current_wires[i+1] + c_out + t_out)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRx'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'CRx'})
                    current_wires[i], current_wires[i+1] = c_out, t_out
                    op_idx += 1
            else:
                for g in ['Rz', 'Ry', 'Rz']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[0] + nxt)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", g))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': g})
                    current_wires[0] = nxt
                    op_idx += 1

        return current_wires, new_indices, new_tensors

class Sim14Ansatz(BaseAnsatz):
    def __init__(self, obmap=set(), layers=1):
        super().__init__(obmap, layers)

    def ansatz(self, current_wires, base_symbol):
        new_indices, new_tensors = [], []
        N = len(current_wires)

        for l in range(self.layers):
            op_idx = 0
            if N > 1:
                for i in range(N):
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'Ry'})
                    current_wires[i] = nxt
                    op_idx += 1
                for i in range(N):
                    c_idx, t_idx = i, (i - 1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRx'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'CRx'})
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1

                for i in range(N):
                    nxt = self.get_char()
                    new_indices.append(current_wires[i] + nxt)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'Ry'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'Ry'})
                    current_wires[i] = nxt
                    op_idx += 1
                for i in range(N, 0, -1):
                    c_idx, t_idx = i % N, (i-1) % N
                    c_out, t_out = self.get_char(), self.get_char()
                    new_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", 'CRx'))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': 'CRx'})
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
            else:
                for g in ['Rx', 'Rz', 'Rx']:
                    nxt = self.get_char()
                    new_indices.append(current_wires[0] + nxt)
                    # new_tensors.append((f"{base_symbol}_l{l}_{op_idx}", g))
                    new_tensors.append({'name': f"{base_symbol}_l{l}_{op_idx}", 'op_type': g})
                    current_wires[0] = nxt
                    op_idx += 1

        return current_wires, new_indices, new_tensors