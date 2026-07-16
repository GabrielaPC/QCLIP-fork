import torch, math
import torch.nn as nn
import torch.nn.functional as F

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

        logits = (text_emb @ image_emb.T) / self.temperature

        loss_t2i = self.cross_entropy(logits, labels)
        loss_i2t = self.cross_entropy(logits.T, labels)
        
        return 0.5 * (loss_t2i + loss_i2t)

# Hilbert-Schmidt InfoNCE loss for QML
class HS_InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)
        text_emb = text_emb.flatten(start_dim=1).to(torch.complex64)
        image_emb = image_emb.flatten(start_dim=1).to(torch.complex64)
        text_emb = F.normalize(text_emb.reshape(B, -1), dim=1, p=2)
        image_emb = F.normalize(image_emb.reshape(B, -1), dim=1, p=2)

        # Vectorization checks & complex-conjugate transposition
        logits = (text_emb @ image_emb.conj().t()).abs() / self.temperature
        
        loss_txt = self.cross_entropy(logits, labels)
        loss_img = self.cross_entropy(logits.T, labels)
        sym_loss = 0.5 * (loss_txt + loss_img)

        # Purity regularization to prevent state collapse in text structures
        if self.lambda_reg > 0 and B > 1:
            txt_overlap = (text_emb @ text_emb.conj().t()).abs()
            mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
            purity_penalty = txt_overlap[mask].mean()
            return sym_loss + self.lambda_reg * purity_penalty
            
        return sym_loss

# Fubini-Study InfoNCE loss for QML
class FS_InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.1, label_smoothing: float = 0.1, eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        B = text_emb.size(0)
        labels = torch.arange(B, device=text_emb.device)
        text_emb = text_emb.flatten(start_dim=1).to(torch.complex64)
        image_emb = image_emb.flatten(start_dim=1).to(torch.complex64)
        text_emb = F.normalize(text_emb.reshape(B, -1), dim=1, p=2)
        image_emb = F.normalize(image_emb.reshape(B, -1), dim=1, p=2)

        # Compute angular similarity matrix via Fubini-Study distance
        overlap = torch.clamp((text_emb @ image_emb.conj().t()).abs(), 0.0, 1.0 - self.eps)
        overlap = 0.5 + (0.5 * overlap)  # Map to [0.5, 1] for stability
        logits = (torch.asin(overlap) / (math.pi / 2)) / self.temperature

        loss_txt = self.cross_entropy(logits, labels)
        loss_img = self.cross_entropy(logits.T, labels)
        sym_loss = 0.5 * (loss_txt + loss_img)

        # Apply angular purity penalty to off-diagonal elements
        if self.lambda_reg > 0 and B > 1:
            txt_overlap = torch.clamp((text_emb @ text_emb.conj().t()).abs(), 0.0, 1.0 - self.eps)
            txt_thetas = torch.acos(txt_overlap)
            txt_logits = 1.0 - (txt_thetas / (math.pi / 2))
            
            mask = ~torch.eye(B, dtype=torch.bool, device=text_emb.device)
            purity_penalty = txt_logits[mask].mean()
            return sym_loss + self.lambda_reg * purity_penalty
            
        return sym_loss