from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch, math
import torch.nn as nn
from opt_einsum import contract_expression
from opt_einsum import parser as oe_parser
from tqdm import tqdm

class TNModel(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.symbols: List[str] = []
        self.shapes: List[Tuple[int, ...]] = []
        self.params = nn.ParameterDict()
        # self.params = nn.ParameterList([])
        # self.sym2param: Dict[str, nn.Parameter] = nn.ParameterDict({})
        self.path_cache = {}
        self.out_dim: int = out_dim
        self.norm_head = nn.LayerNorm(out_dim)
    
    @property
    def pcount(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def grad_norm(self):
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    

    def from_symbols(self, tn_arr):
        symbols_hash = set(self.symbols)

        for tensor_arr in tqdm(tn_arr):
            flat_tensors = self._flatten_symbols(tensor_arr)
            for symbol_info in flat_tensors:
                symbol = symbol_info['name']
                if symbol not in symbols_hash:
                    symbols_hash.add(symbol)
                    self.symbols.append(symbol)
                    self.shapes.append(tuple(symbol_info['shape']))

        self.init_params()

    def _flatten_symbols(self, val):
        if isinstance(val, dict) and 'name' in val:
            return [val]
        elif isinstance(val, list):
            res = []
            for item in val:
                res.extend(self._flatten_symbols(item))
            return res
        return []

    def init_params(self):
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through ')
        
        # self.params = nn.ParameterList([nn.Parameter(torch.empty(shape)) for shape in self.shapes])
        # self.sym2param = nn.ParameterDict({sym: param for sym, param in zip(self.symbols, self.params)})
        for symbol, shape in zip(self.symbols, self.shapes):
            param = nn.Parameter(torch.empty(shape))
            order = len(shape)

            if order == 0:
                nn.init.constant_(param, 1.0)
            elif order == 1:
                nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(shape[0]))
            else:
                fan_out = shape[0]
                fan_in = math.prod(shape[1:])
                std = math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(param, mean=0.0, std=std)
            self.params[symbol] = param

    def compile_batch(self, batch: List[tuple[str, List[str]]]) -> torch.Tensor:
        groups = defaultdict(list)
        for einsum_expr, tensor_arr in batch:
            shapes, _ = zip(*[(tuple(sym_dict['shape']), sym_dict['name']) for sym_dict in tensor_arr])
            groups[(einsum_expr, shapes)].append(tensor_arr)
        
        for (einsum_expr, _), tensor_arr in groups.items():
            minibatch_size = len(tensor_arr)
            tensor_columns = zip(*tensor_arr)
            shapes = []
            for col in tensor_columns:
                shapes.append(torch.Size([minibatch_size, *col[0]['shape']]))

            input_subscripts, output_subscript = einsum_expr.split('->')
            batch_symbol = next(oe_parser.gen_unused_symbols(einsum_expr, 1))
            batched_inputs = [batch_symbol + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->{batch_symbol}{output_subscript}"

            cache_key = (batched_str, tuple(shapes))
            if cache_key not in self.path_cache: 
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)

    def forward(self, batch: List[tuple[str, List[str]]]) -> torch.Tensor:
        groups = defaultdict(list)
        for i, (einsum_expr, tensor_arr) in enumerate(batch):
            shapes, _ = zip(*[(tuple(sym_dict['shape']), sym_dict['name']) for sym_dict in tensor_arr])
            groups[(einsum_expr, shapes)].append((i, tensor_arr))

        output = torch.zeros(len(batch), self.out_dim, device=next(self.params.parameters()).device)
        for (einsum_expr, _), items in groups.items():
            indices, symbols_arr =  zip(*items)
            symbol_columns = zip(*symbols_arr)
            shapes, tensors = [], []
            for col in symbol_columns:
                tensor_batch = torch.stack([self.params[sym_dict['name']] for sym_dict in col], dim=0)
                shapes.append(tensor_batch.shape)
                tensors.append(tensor_batch)
            
            input_subscripts, output_subscript = einsum_expr.split('->')
            batch_symbol = next(oe_parser.gen_unused_symbols(einsum_expr, 1))
            batched_inputs = [batch_symbol + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->{batch_symbol}{output_subscript}"

            cache_key = (batched_str, tuple(shapes))
            if cache_key not in self.path_cache:
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)
            output[list(indices)] = self.path_cache[cache_key](*tensors)
        
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)