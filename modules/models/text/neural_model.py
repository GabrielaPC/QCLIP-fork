from typing import List, Union
import torch
import torch.nn as nn
from tqdm import tqdm

class WordMLP(nn.Module):
    """Dynamically sized gating logic for compositional words."""
    def __init__(self, num_inputs: int, hdim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs * hdim, hdim * 2),
            nn.LayerNorm(hdim * 2),
            nn.GELU(),
            nn.Linear(hdim * 2, hdim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPModel(nn.Module):
    def __init__(self, hdim: int, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.hdim = hdim
        self.leaves = nn.ParameterDict()
        self.mlps = nn.ModuleDict()
        self.final_projection = nn.Linear(hdim, out_dim)
        self.norm_head = nn.LayerNorm(out_dim)

    def from_plans(self, plans_arr: List[List[dict]]):
        unique_leaves = set()
        unique_mlps = {}
        target_device = next(self.parameters()).device if list(self.parameters()) else 'cuda'

        for row in tqdm(plans_arr):
            is_nested = isinstance(row, list) and len(row) > 0 and isinstance(row[0], list)
            plans = row if is_nested else [row]
            for plan in plans:
                for node in plan:
                    symbol = node['name']
                    inputs_required = len(node['idx']) - 1
                    
                    if inputs_required == 0:
                        unique_leaves.add(symbol)
                    else:
                        unique_mlps[symbol] = inputs_required
        # temp_mlps[sym] = WordMLP(inputs_required, self.dim)
        # temp_leaves[sym] = nn.Parameter(torch.randn(self.dim, device='cpu') * 0.02)
        temp_leaves, temp_mlps = {}, {}
        for symbol in tqdm(unique_leaves, desc="Allocating Leaf Parameters"):
            temp_leaves[symbol] = nn.Parameter(torch.randn(self.hdim) * 0.02)
        for key, inputs_required in tqdm(unique_mlps.items(), desc="Allocating WordMLP Modules"):
            temp_mlps[key] = WordMLP(inputs_required, self.hdim)
        self.leaves = nn.ParameterDict(temp_leaves)
        self.mlps = nn.ModuleDict(temp_mlps)

    def forward(self, batch: List[Union[dict, tuple]]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_outputs = []

        for plan, _ in batch:
            max_wire = max(max(node['idx']) for node in plan) if plan else 0
            wires = [None] * (max_wire + 1)

            for node in plan:
                sym = node['name']
                out_wire = node['out']
                inputs_required = len(node['idx']) - 1

                if inputs_required == 0:
                    # Leaf fetch with Out-Of-Vocabulary fallback (critical for eval sets)
                    wires[out_wire] = self.leaves.get(sym, torch.zeros(self.hdim, device=device))
                else:
                    in_vectors = []
                    for idx in node['idx']:
                        if idx != out_wire:
                            # Pull from wire, fallback to zero if structural parse error caused a dead wire
                            vec = wires[idx] if wires[idx] is not None else torch.zeros(self.hdim, device=device)
                            in_vectors.append(vec)

                    if sym in self.mlps:
                        x = torch.cat(in_vectors, dim=0)
                        wires[out_wire] = self.mlps[sym](x)
                    else:
                        wires[out_wire] = torch.zeros(self.hdim, device=device)

            # Safety fallback for catastrophic parser failures returning empty trees
            final_vector = wires[0] if (wires and wires[0] is not None) else torch.zeros(self.hdim, device=device)
            batch_outputs.append(final_vector)

        output = torch.stack(batch_outputs, dim=0)
        output = self.final_projection(output)
        output = self.norm_head(output)
        return torch.nn.functional.normalize(output, p=2, dim=-1, eps=1e-12)