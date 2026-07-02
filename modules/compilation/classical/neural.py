from collections import defaultdict
from itertools import count
from tqdm import tqdm

class MLPCompiler:
    def __init__(self):
        self.id = type(self).__name__

    def compile_tn(self, tn):
        index_to_words = defaultdict(list)
        for i, (word, idx_arr, type_arr) in enumerate(tn):
            for idx in idx_arr:
                index_to_words[idx].append(i)

        root_idx = next((i for i, (_, idx_arr, _) in enumerate(tn) if 0 in idx_arr), 0)

        queue = [(root_idx, 0)]
        visited_words = {root_idx}
        word_outputs = {}

        while queue:
            curr_word_idx, out_wire = queue.pop(0)
            word_outputs[curr_word_idx] = out_wire
            word, idx_arr, type_arr = tn[curr_word_idx]
            for idx in idx_arr:
                if idx != out_wire:
                    for nxt_word_idx in index_to_words[idx]:
                        if nxt_word_idx not in visited_words:
                            visited_words.add(nxt_word_idx)
                            queue.append((nxt_word_idx, idx))   

        plan = []
        for w_idx in reversed(list(word_outputs.keys())):
            word, idx_arr, type_arr = tn[w_idx]
            symbol_name = f"{word}_{'@'.join(type_arr)}"
            plan.append({
                'name': symbol_name,
                'idx': idx_arr,
                'out': word_outputs[w_idx]
            })

        return plan
    
    def compile_tn_v2(self, tn):
        index_to_words = defaultdict(list)
        for i, (word, idx_arr, type_arr) in enumerate(tn):
            for idx in idx_arr:
                index_to_words[idx].append(i)

        root_idx = next((i for i, (_, idx_arr, _) in enumerate(tn) if 0 in idx_arr), 0)

        queue = [(root_idx, 0)]
        visited_words = {root_idx}
        word_outputs = {}

        idx_set = set()
        while queue:
            curr_word_idx, out_wire = queue.pop(0)
            word_outputs[curr_word_idx] = out_wire
            word, idx_arr, type_arr = tn[curr_word_idx]
            for idx in idx_arr:
                idx_set.add(idx)
                if idx != out_wire:
                    for nxt_word_idx in index_to_words[idx]:
                        if nxt_word_idx not in visited_words:
                            visited_words.add(nxt_word_idx)
                            queue.append((nxt_word_idx, idx))   
        
        counter = count(start=ord(max(idx_set)) + 1)
        plan = []
        for w_idx in reversed(list(word_outputs.keys())):
            word, idx_arr, type_arr = tn[w_idx]
            if len(idx_arr) == 1:
                plan.append({
                    'name': word,
                    'idx': idx_arr,
                    'out': word_outputs[w_idx]
                })
            else:
                new_idx = chr(next(counter))
                plan.append({
                    'name': word,
                    'idx': [new_idx],
                    'out': [new_idx]
                })
                plan.append({
                    'name': '@'.join(type_arr),
                    'idx': idx_arr + [new_idx],
                    'out': word_outputs[w_idx]
                })
        return plan
    
    def compile_dataset(self, tn_df):
        # blueprint_df = pd.DataFrame(index=tn_df.index)
        tn_cols = [col for col in tn_df.columns if col.endswith('_diagram')]
        
        for col in tn_cols:
            base_label = col.replace('_diagram', '')
            first_val = tn_df[col].iloc[0]
            
            is_nested = isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], list)

            plans_arr = []
            symbols_arr = []
            for row in tqdm(tn_df[col], desc=f"Compiling {base_label}"):
                if is_nested:
                    results = [self.compile_tn(t) for t in row]
                    plans_arr.append(results)
                    row_symbols = [[step['name'] for step in plan] for plan in results]
                    symbols_arr.append(row_symbols)
                else:
                    plan = self.compile_tn(row)
                    plans_arr.append(plan)
                    row_symbols = [step['name'] for step in plan]
                    symbols_arr.append(row_symbols)
                    
            tn_df[f"{base_label}_einsum"] = plans_arr
            tn_df[f"{base_label}_symbols"] = symbols_arr
            
        return tn_df