import torch, math
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
class DB_Sqrt(nn.Module):
    def __init__(self, iters: int = 6, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        orig_shape = A.shape
        D = orig_shape[-1]
        
        A_flat = A.reshape(-1, D, D)
        B = A_flat.size(0)
        device, dtype = A.device, A.dtype

        A_flat = complex_nan_to_num(A_flat, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. Hermitian Symmetrization: A = 0.5 * (A + A^H)
        A_flat = 0.5 * (A_flat + A_flat.conj().transpose(-2, -1))

        # 3. Trace Conditioning: Scale matrix to unit trace to optimize convergence rate
        tr_A = torch.real(torch.einsum('bii->b', A_flat)).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
        A_scaled = A_flat / tr_A

        # 4. Diagonal Ridge Jitter: Prevents singularity for zero/degenerate eigenvalues
        I = torch.eye(D, dtype=dtype, device=device).unsqueeze(0).expand(B, D, D)
        
        # Spectrum-splitting jitter to resolve degenerate spectra during backward pass
        splitting_vec = torch.linspace(self.eps, self.eps * 10, D, device=device, dtype=A_flat.real.dtype)
        jitter = torch.diag_embed(splitting_vec).to(dtype).unsqueeze(0).expand(B, D, D)
        
        Y = A_scaled + jitter
        Z = I.clone()

        # 5. Coupled Denman-Beavers Iterations
        for _ in range(self.iters):
            Y_inv = torch.linalg.inv(Y)
            Z_inv = torch.linalg.inv(Z)

            Y = 0.5 * (Y + Z_inv)
            Z = 0.5 * (Z + Y_inv)

            # Re-enforce Hermitian symmetry at each step to suppress floating-point drift
            Y = 0.5 * (Y + Y.conj().transpose(-2, -1))
            Z = 0.5 * (Z + Z.conj().transpose(-2, -1))

        # 6. Un-scale back to original tensor magnitude: sqrt(A) = sqrt(Tr(A)) * Y
        sqrt_A = Y * torch.sqrt(tr_A)
        
        return sqrt_A.reshape(orig_shape)

class NS_Sqrt(nn.Module):
    def __init__(self, iters: int = 8, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        orig_shape = A.shape
        D = orig_shape[-1]
        A_flat = A.reshape(-1, D, D)
        B = A_flat.size(0)
        device, dtype = A.device, A.dtype

        # 1. Sanitize & Symmetrize
        A_flat = complex_nan_to_num(A_flat)
        A_flat = 0.5 * (A_flat + A_flat.conj().transpose(-2, -1))

        tr_A = torch.real(torch.einsum('bii->b', A_flat)).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
        A_norm = A_flat / tr_A

        # 2. Diagonal Ridge Jitter
        I = torch.eye(D, dtype=dtype, device=device).unsqueeze(0).expand(B, D, D)
        splitting_vec = torch.linspace(self.eps, self.eps * 10, D, device=device, dtype=A_flat.real.dtype)
        jitter = torch.diag_embed(splitting_vec).to(dtype).unsqueeze(0).expand(B, D, D)
        A_reg = A_norm + jitter

        # 3. Frobenius Norm Normalization (Guarantees Newton-Schulz Convergence)
        norm_A = A_reg.norm(p='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        norm_A_c = norm_A.to(dtype)

        Y = A_reg / norm_A_c
        Z = I.clone()

        # 4. Coupled Newton-Schulz Iterations
        for _ in range(self.iters):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z

            # Re-enforce Hermitian symmetry at each step to suppress floating-point drift
            Y = 0.5 * (Y + Y.conj().transpose(-2, -1))
            Z = 0.5 * (Z + Z.conj().transpose(-2, -1))

        # 5. Rescale back: sqrt(A) = Y * sqrt(||A||_F)
        sqrt_A = Y * torch.sqrt(norm_A_c * tr_A)
        return sqrt_A.reshape(orig_shape)

class ExactSpectralMatrixSqrt(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        orig_shape = A.shape
        D = orig_shape[-1]
        A_flat = A.reshape(-1, D, D)
        B = A_flat.size(0)
        orig_device = A.device

        # 1. Sanitize & Hermitian symmetrization
        A_flat = complex_nan_to_num(A_flat)
        A_flat = 0.5 * (A_flat + A_flat.conj().transpose(-2, -1))

        # 2. Spectrum-splitting jitter
        splitting_vec = torch.linspace(self.eps, self.eps * 100, D, device=orig_device, dtype=A_flat.real.dtype)
        jitter = torch.diag_embed(splitting_vec).to(A.dtype).unsqueeze(0).expand(B, D, D)
        A_reg = A_flat + jitter

        A_cpu = A_reg.cpu()

        try:
            evals, evecs = torch.linalg.eigh(A_cpu)
            evals = torch.clamp(evals, min=self.eps)
            sqrt_evals = torch.diag_embed(torch.sqrt(evals)).to(A.dtype)
            sqrt_A = evecs @ sqrt_evals @ evecs.conj().transpose(-2, -1)
        except torch._C._LinAlgError:
            U, S, Vh = torch.linalg.svd(A_cpu)
            sqrt_S = torch.diag_embed(torch.sqrt(torch.clamp(S, min=self.eps))).to(A.dtype)
            sqrt_A = U @ sqrt_S @ Vh

        return sqrt_A.to(orig_device).reshape(orig_shape)



def complex_nan_to_num(z: torch.Tensor, nan: float = 0.0, posinf: float = 1.0, neginf: float = -1.0) -> torch.Tensor:
    real_part = torch.nan_to_num(z.real, nan=nan, posinf=posinf, neginf=neginf)
    imag_part = torch.nan_to_num(z.imag, nan=nan, posinf=posinf, neginf=neginf)
    return torch.complex(real_part, imag_part)


class BaseInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def describe(self) -> None:
        hyperparams = self.get_hyperparams()
        param_str = " | ".join([
            f"| {k}: {v:.4f} " if isinstance(v, float) else f"| {k}: {v} " 
            for k, v in hyperparams.items()
        ])
        print(f"{self.__class__.__name__} {param_str}")

    def get_hyperparams(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "label_smoothing": self.label_smoothing,
        }

    def _ensure_density_matrices(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B = text_emb.size(0)
        if text_emb.dim() == 2:
            D = int(math.sqrt(text_emb.size(1)))
            rho_text = text_emb.view(B, D, D).to(torch.complex64)
            rho_image = image_emb.view(B, D, D).to(torch.complex64)
        else:
            D = text_emb.size(-1)
            rho_text = text_emb.to(torch.complex64)
            rho_image = image_emb.to(torch.complex64)
        return rho_text, rho_image, D

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_similarity().")

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=text_emb.device)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)

        # 1. Compute similarity matrix [B, B]
        similarity = self.compute_similarity(text_emb, image_emb)

        # 2. Scale by temperature
        logits = similarity / self.temperature

        # 3. Symmetrized Cross-Entropy Loss
        loss_t2i = self.cross_entropy(logits, labels)
        loss_i2t = self.cross_entropy(logits.T, labels)
        sym_loss = 0.5 * (loss_t2i + loss_i2t)

        # 4. Optional Regularization
        reg_loss = self.compute_regularization(text_emb, image_emb)

        return sym_loss + reg_loss

class InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.describe()

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_flat = text_emb.flatten(start_dim=1)
        image_flat = image_emb.flatten(start_dim=1)
        return (text_flat @ image_flat.conj().T).abs()

class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)
        text_emb = text_emb.flatten(start_dim=1)
        image_emb = image_emb.flatten(start_dim=1)

        logits = (text_emb @ image_emb.conj().T).abs() / self.temperature

        loss_t2i = self.cross_entropy(logits, labels)
        loss_i2t = self.cross_entropy(logits.T, labels)
        
        return 0.5 * (loss_t2i + loss_i2t)

class HS_InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1, label_smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.lambda_reg = lambda_reg
        self.eps = eps

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"lambda_reg": self.lambda_reg, "eps": self.eps})
        return hp

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_flat = F.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        image_flat = F.normalize(image_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        return (text_flat @ image_flat.conj().T).abs()

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        if self.lambda_reg > 0 and B > 1:
            text_flat = F.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
            txt_overlap = torch.clamp((text_flat @ text_flat.conj().T).abs(), 0.0, 1.0 - self.eps)
            txt_thetas = torch.acos(txt_overlap)
            txt_logits = 1.0 - (txt_thetas / (math.pi / 2))
            
            mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
            return self.lambda_reg * txt_logits[mask].mean()
        return torch.tensor(0.0, device=text_emb.device)


class MHS_InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        rho_text, rho_image, D = self._ensure_density_matrices(text_emb, image_emb)
        raw_overlap = torch.real(torch.einsum('imn,jnm->ij', rho_text, rho_image))
        return raw_overlap - (1.0 / D)

class FS_InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1, label_smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.lambda_reg = lambda_reg
        self.eps = eps

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"lambda_reg": self.lambda_reg, "eps": self.eps})
        return hp

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_flat = F.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
        image_flat = F.normalize(image_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)

        overlap = torch.clamp((text_flat @ image_flat.conj().T).abs(), 0.0, 1.0 - self.eps)
        overlap_scaled = 0.5 + (0.5 * overlap)
        return torch.asin(overlap_scaled) / (math.pi / 2)

    def compute_regularization(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        if self.lambda_reg > 0 and B > 1:
            text_flat = F.normalize(text_emb.flatten(start_dim=1).to(torch.complex64), p=2, dim=1)
            txt_overlap = torch.clamp((text_flat @ text_flat.conj().T).abs(), 0.0, 1.0 - self.eps)
            txt_thetas = torch.acos(txt_overlap)
            txt_logits = 1.0 - (txt_thetas / (math.pi / 2))
            
            mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
            return self.lambda_reg * txt_logits[mask].mean()
        return torch.tensor(0.0, device=text_emb.device)

class UJ_InfoNCE(BaseInfoNCELoss):
    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.0, eps: float = 1e-7):
        super().__init__(temperature=temperature, label_smoothing=label_smoothing)
        self.eps = eps
        self.sqrt_module = NS_Sqrt(iters=25, eps=1e-6)

    def get_hyperparams(self) -> Dict[str, Any]:
        hp = super().get_hyperparams()
        hp.update({"eps": self.eps, 
                   "sqrt_engine": self.sqrt_module.__class__.__name__})
        return hp

    def compute_similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        rho_text, rho_image, D = self._ensure_density_matrices(text_emb, image_emb)
        B = rho_text.size(0)

        # 1. Compute exact sqrt(rho_text) -> [B, D, D]
        sqrt_rho_text = self.sqrt_module(rho_text)

        # 2. Form transition matrix M_ij = sqrt(rho_i) @ sigma_j @ sqrt(rho_i) -> [B, B, D, D]
        M = sqrt_rho_text.unsqueeze(1) @ rho_image.unsqueeze(0) @ sqrt_rho_text.unsqueeze(1)
        M = 0.5 * (M + M.conj().transpose(-2, -1))

        # 3. Sum sqrt singular values of M
        sqrt_M = self.sqrt_module(M.reshape(B * B, D, D)).reshape(B, B, D, D)
        tr_sqrt_M = torch.real(torch.diagonal(sqrt_M, dim1=-2, dim2=-1).sum(dim=-1))

        # 4. Final fidelity
        return torch.clamp(tr_sqrt_M ** 2, 0.0, 1.0)

        # overlap_scaled = 0.5 + (0.5 * torch.clamp(tr_sqrt_M, 0.0, 1.0 - self.eps))
        # return torch.asin(overlap_scaled) / (math.pi / 2) 