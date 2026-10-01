from __future__ import annotations
import pandas as pd
import opt_einsum as oe
from itertools import count
from tqdm import tqdm
    
class TNCompiler: 
    def __init__(self, obmap: dict[str, int], decomp_fn):
        self.obmap = obmap
        self.word_dim = obmap.get('sem', 16)
        self.decomposition = decomp_fn
        self.id = type(self).__name__ + '_' + str(obmap['n']) + '_' + str(obmap['s']) + '_' + str(obmap['p']) + '_' + str(obmap['out'])

    def compile_tn(self, tn):
        all_indices = [idx for _, idx_arr, _ in tn for idx in idx_arr]
        max_idx = max(all_indices) if all_indices else 0
        index_counter = count(start=max_idx + 1)

        processed_tensors = []

        for word, idx_arr, type_arr in tn:
            if len(idx_arr) != len(type_arr):
                raise ValueError(f"Mismatch between indices and types for word '{word}' in sentence '{' '.join([w for w, _, _ in tn])}': "
                                 f"indices={idx_arr}, types={type_arr}")
            dims = [self.obmap.get(t, 1) for t in type_arr]
            symbol = word + '_' + '@'.join(type_arr)
            if self.decomposition is None:
                processed_tensors.append((symbol, idx_arr, tuple(dims)))
            else:
                cores = self.decomposition.decompose(symbol, idx_arr, dims, index_counter)
                processed_tensors.extend(cores)

        input_subs = []
        symbols = []

        for symbol, idx_arr, shape in processed_tensors:
            subs = "".join(oe.parser.get_symbol(i) for i in idx_arr)
            input_subs.append(subs)
            symbols.append({'name': symbol, 'shape': list(shape)})

        output_subs = oe.parser.get_symbol(0)
        einsum_str = f"{','.join(input_subs)}->{output_subs}"
        return einsum_str, symbols

    def compile_decoupled_tn(self, tn):
        all_indices = [idx for _, idx_arr, _ in tn for idx in idx_arr]
        max_idx = max(all_indices) if all_indices else 0
        index_counter = count(start=max_idx + 1)

        processed_tensors = []

        for word, idx_arr, type_arr in tn:
            if len(idx_arr) != len(type_arr):
                raise ValueError(
                    f"Mismatch between indices and types for word '{word}' in sentence "
                    f"'{' '.join([w for w, _, _ in tn])}': indices={idx_arr}, types={type_arr}")

            connect_idx = next(index_counter)

            word_symbol = word
            word_indices = [connect_idx]
            word_shape = (self.word_dim,)
            processed_tensors.append((word_symbol, word_indices, word_shape))

            type_symbol = '@'.join(type_arr)
            type_indices = list(idx_arr) + [connect_idx]
            type_dims = [self.obmap.get(t, 1) for t in type_arr] + [self.word_dim]

            if self.decomposition is None:
                processed_tensors.append((type_symbol, type_indices, tuple(type_dims)))
            else:
                cores = self.decomposition.decompose(type_symbol, type_indices, type_dims, index_counter)
                processed_tensors.extend(cores)

            input_subs = []
            symbols = []

        for symbol, idx_arr, shape in processed_tensors:
            subs = "".join(oe.parser.get_symbol(i) for i in idx_arr)
            input_subs.append(subs)
            symbols.append({'name': symbol, 'shape': list(shape)})

        output_subs = oe.parser.get_symbol(0)
        einsum_str = f"{','.join(input_subs)}->{output_subs}"
        return einsum_str, symbols
    
    def compile_dataset(self, tn_df):
        # blueprint_df = pd.DataFrame(index=tn_df.index)
        tn_cols = [col for col in tn_df.columns if col.endswith('_diagram')]
        for col in tn_cols:
            base_label = col.replace('_diagram', '')
            einsum_col = f"{base_label}_einsum"
            symbols_col = f"{base_label}_symbols"

            first_val = tn_df[col].iloc[0]
            is_nested_list = (isinstance(first_val, list) 
                              and len(first_val) > 0 
                              and isinstance(first_val[0], list)
                             )
            einsum_arr, symbols_arr = [], []

            for row_val in tqdm(tn_df[col]):
                if is_nested_list:
                    row_einsums, row_symbols = [], []
                    for tn in row_val:
                        e_str, syms = self.compile_tn(tn)
                        row_einsums.append(e_str)
                        row_symbols.append(syms)
                    einsum_arr.append(row_einsums)
                    symbols_arr.append(row_symbols)
                else:
                    e_str, syms = self.compile_tn(row_val)
                    einsum_arr.append(e_str)
                    symbols_arr.append(syms)
            tn_df[einsum_col] = einsum_arr
            tn_df[symbols_col] = symbols_arr
        return tn_df
    