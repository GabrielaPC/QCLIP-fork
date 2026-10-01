import torch.nn as nn
from itertools import count
import torch, math
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from opt_einsum import contract_expression
from modules.compilation.quantum.gates import *

class QuantumFeatureMap(nn.Module):
    def __init__(self, k: int, layers: int, batch_size: int, out_dim: int = None, id_init=False, method='mlp', discard=False):
        super().__init__()
        self.k = k
        self.out_dim = 2 ** k if out_dim is None else out_dim
        self.layers = layers
        self.batch_size = batch_size
        self.params = nn.ParameterList([])
        self.sym2param = {}
        self.method = method
        self.discard = discard

        if self.method == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.LayerNorm(self.out_dim // 2),
                nn.SiLU(),
                nn.Linear(self.out_dim // 2, 3 * k * layers),
            )
            nn.init.uniform_(self.projector[3].weight, a=-1e-4, b=1e-4)
            nn.init.zeros_(self.projector[3].bias)
        if self.method == 'pca':
            self.pca = IncrementalPCA(n_components= 3 * k * layers)

        self.compile_fmap()
        self.init_params(id_init)
        self.global_pca_max = 1.0  # Placeholder for normalization during PCA projection

    def get_device(self):
        return next(self.parameters()).device

    def init_params(self, id_init=False):
        num_params = len(self.sym2param)
        if id_init:
            param_data = [torch.randn(1) * 0.01 for _ in range(num_params)]
        else:
            param_data = [torch.randn(1) * 2 * torch.pi for _ in range(num_params)]
        self.params = nn.ParameterList([nn.Parameter(p, requires_grad=True) for p in param_data])

    def reset_char(self):
        self.char_idx = count(0)

    def get_char(self):
        i = next(self.char_idx)
        chars = "acdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return chars[i] if i < len(chars) else chr(192 + i - len(chars))
    
    def _get_params(self):
        return {sym: float(self.params[idx].detach().cpu().item()) for sym, idx in self.sym2param.items()}

    def compile_fmap(self):
        self.reset_char()
        input_indices = []
        gate_arr = []
        shape_arr = []

        current_wires = [self.get_char() for _ in range(self.k)]
        input_indices.extend([w for w in current_wires])
        gate_arr.extend({'name': None, 'op_type': '0'} for _ in range(self.k))
        shape_arr.extend([[2]] * self.k)

        symbol_idx = 0
        for l in range(self.layers):
            op_idx = 0

            for i in range(self.k):
                for rotation in ['Rx', 'Ry', 'Rz']:
                    nxt = self.get_char()
                    input_indices.append('b' + current_wires[i] + nxt)
                    symbol = f"ftr_{rotation}_l{l}_{i}"
                    gate_arr.append({'name': symbol, 'op_type': rotation})
                    shape_arr.append([self.batch_size, 2, 2])
                    current_wires[i] = nxt

            for i in range(self.k):
                nxt = self.get_char()
                input_indices.append(current_wires[i] + nxt)
                symbol = f"img_Ry_l{l}_{op_idx}"
                gate_arr.append({'name': symbol, 'op_type': 'Ry'})
                self.sym2param[symbol] = symbol_idx
                symbol_idx += 1
                shape_arr.append([2, 2])
                current_wires[i] = nxt
                op_idx += 1

            if self.k > 1:
                for i in range(self.k):
                    c_idx, t_idx = i, (i + 1) % self.k
                    c_out, t_out = self.get_char(), self.get_char()
                    input_indices.append(current_wires[c_idx] + current_wires[t_idx] + c_out + t_out)
                    symbol = f"img_CRz_l{l}_{op_idx}"
                    gate_arr.append({'name': symbol, 'op_type': 'CRz'})
                    self.sym2param[symbol] = symbol_idx
                    symbol_idx += 1
                    shape_arr.append([2, 2, 2, 2])
                    current_wires[c_idx], current_wires[t_idx] = c_out, t_out
                    op_idx += 1
        
        einsum_str = f"{','.join(input_indices)}->b{''.join(current_wires)}"
        self.gate_arr = gate_arr
        self.einsum_expr = einsum_str
        self.contraction_path = contract_expression(einsum_str, *shape_arr)

    def fit_image_pca(self, image_stream, batch_size=2048):
        batch = []
        for img_tensor in image_stream:
            batch.append(img_tensor.cpu().numpy() if hasattr(img_tensor, 'numpy') else img_tensor)
            if len(batch) == batch_size:
                self.pca.partial_fit(np.array(batch))
                batch = []
        if batch:
            self.pca.partial_fit(np.array(batch))
        # Update the global PCA maximum after fitting
        self.global_pca_max = np.max(np.abs(self.pca.transform(np.array(batch))))

    def get_features(self, img_vecs):
        if self.method == 'mlp':
            img_vecs = torch.as_tensor(img_vecs, dtype=torch.float32, device=self.get_device())
            features = self.projector(img_vecs.view(img_vecs.shape[0], -1)) * torch.pi
        if self.method == 'pca':
            if torch.is_tensor(img_vecs):
                flat_vector = img_vecs.detach().cpu().numpy().reshape(len(img_vecs), -1)
            else:
                flat_vector = np.asarray(img_vecs).reshape(len(img_vecs), -1)
            raw_pca = self.pca.transform(flat_vector)
            norm_pca = (raw_pca / (self.global_pca_max + 1e-8)) * np.pi
            features = torch.tensor(norm_pca, dtype=torch.float32, device=self.get_device())
        return features 
    
    def encode_features(self, img_vec):
        img_vec = torch.as_tensor(img_vec, dtype=torch.float32, device=self.get_device())
        is_1d = (img_vec.ndim == 1)
        if is_1d: img_vec = img_vec.unsqueeze(0)
        features = self.get_features(img_vec)
        if is_1d: features = features.squeeze(0)
        axis_map = {'Rx': 0, 'Ry': 1, 'Rz': 2}
        sym2ftr = {}
        for item in self.gate_arr:
            symbol, gate = item['name'], item['op_type']
            if gate == '0': continue
            parts = symbol.split('_')
            param_type = parts[0]
            if param_type == 'ftr':
                layer_idx = int(parts[2][1:])
                qubit_idx = int(parts[3])
                axis_offset = axis_map[gate]
                flat_idx = (layer_idx * self.k * 3) + (qubit_idx * 3) + axis_offset 

                feature_tensor = features[flat_idx]
                sym2ftr[symbol] = feature_tensor
        return sym2ftr

    def forward(self, img_vecs):
        features = self.get_features(img_vecs)
        thetas = torch.cat([p for p in self.params])
        tensor_arr = []
        axis_map = {'Rx': 0, 'Ry': 1, 'Rz': 2}
        for item in self.gate_arr:
            symbol, gate = item['name'], item['op_type']
            if gate == '0':
                tensor_arr.append(torch.tensor([1, 0], dtype=torch.complex64, device=self.get_device()))
                continue

            parts = symbol.split('_')
            param_type = parts[0]

            if param_type == 'ftr':
                layer_idx = int(parts[2][1:])
                qubit_idx = int(parts[3])
                axis_offset = axis_map[gate]
                flat_idx = (layer_idx * self.k * 3) + (qubit_idx * 3) + axis_offset 

                feature_tensor = features[:, flat_idx]
                if gate == 'Rx': tensor_arr.append(Rx(feature_tensor))
                if gate == 'Ry': tensor_arr.append(Ry(feature_tensor))
                if gate == 'Rz': tensor_arr.append(Rz(feature_tensor))
            elif param_type == 'img':
                if gate == 'Ry':
                    idx = self.sym2param[symbol]
                    tensor_arr.append(Ry(thetas[idx]))
                if gate == 'CRz':
                    idx = self.sym2param[symbol]
                    tensor_arr.append(CRz(thetas[idx]))

        if self.discard:
            pure_state = self.contraction_path(*tensor_arr)
            batch_dim = pure_state.shape[0]
            pure_state = pure_state.reshape(batch_dim, 2**self.k, -1)
            rho = torch.bmm(pure_state, pure_state.conj().transpose(1, 2))
            return rho
        else:
            return self.contraction_path(*tensor_arr)   


class QFMap_CPTP(nn.Module):
    def __init__(self, base_image_model: nn.Module, gamma: float = 0.3):
        super().__init__()
        self.image_model = base_image_model
        self.gamma = gamma

    # --- Delegate Einsum Attributes & Methods to Base Feature Map ---
    @property
    def einsum_expr(self):
        return getattr(self.image_model, 'einsum_expr', None)

    @property
    def gate_arr(self):
        return getattr(self.image_model, 'gate_arr', None)

    def describe_einsum(self, *args, **kwargs):
        if hasattr(self.image_model, 'describe_einsum'):
            return self.image_model.describe_einsum(*args, **kwargs)
        raise AttributeError(f"Base model {self.image_model.__class__.__name__} does not implement describe_einsum.")

    def __getattr__(self, name: str):
        # Fallback to base_image_model for any unhandled attributes (e.g., sym2param, fit_image_pca)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.image_model, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_model(x)
        
        orig_shape = img_emb.shape
        B = img_emb.size(0)
        
        if img_emb.dim() == 2:
            D = int(math.sqrt(img_emb.size(1)))
            rho = img_emb.view(B, D, D).to(torch.complex64)
        else:
            D = img_emb.size(-1)
            rho = img_emb.to(torch.complex64)

        # Construct maximally mixed identity matrix 1/d * I_d
        I_d = torch.eye(D, dtype=torch.complex64, device=img_emb.device).unsqueeze(0).expand(B, D, D)
        rho_mixed = (1.0 / D) * I_d

        # Apply Depolarizing Channel: (1 - gamma) * rho + gamma * (1/d * I)
        rho_out = (1.0 - self.gamma) * rho + self.gamma * rho_mixed

        if len(orig_shape) == 2:
            return rho_out.view(B, D * D)
        return rho_out