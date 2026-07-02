import torch
from typing import Callable
import torch.nn.functional as F

mscoco_mapper = lambda batch: (batch["image"], batch["caption"])
aro_mapper = lambda batch: (batch["image"], batch["pos_caption"])
svo_mapper = lambda batch: (batch["pos_image"], batch["caption"])

class ContrastiveTrainer:
    def __init__(self, image_model, text_model, optimizer, loss_fn, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader, batch_mapper: Callable) -> float:
        self.image_model.train()
        self.text_model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            images, texts = batch_mapper(batch)
            images = images.to(self.device)

            image_emb = self.image_model(images)
            text_emb = self.text_model(texts)

            loss = self.loss_fn(text_emb, image_emb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # DIAGNOSTICS
            # text_grads = [p.grad.norm().item() for p in self.text_model.parameters() if p.grad is not None]
            # image_grads = [p.grad.norm().item() for p in self.image_model.parameters() if p.grad is not None]
            # print(f"Active Text Param Groups: {len(text_grads)} | Mean Grad Norm: {sum(text_grads)/len(text_grads) if text_grads else 0}")
            # print(f"Active Image Param Groups: {len(image_grads)} | Mean Grad Norm: {sum(image_grads)/len(image_grads) if image_grads else 0}")
            # print(f"Text Emb Norm Mean: {text_emb.norm(dim=-1).mean().item():.4f}")
            # print(f"Image Emb Norm Mean: {image_emb.norm(dim=-1).mean().item():.4f}")

            # torch.nn.utils.clip_grad_norm_(self.text_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item() * images.shape[0]
        
        return epoch_loss / len(dataloader.dataset)

class MMEvaluator:
    def __init__(self, image_model, text_model, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.device = device

    @torch.no_grad()
    def _encode_txt(self, texts):
        return F.normalize(self.text_model(texts).flatten(1), dim=1)
    
    @torch.no_grad()
    def _encode_img(self, images):
        return F.normalize(self.image_model(images.to(self.device)).flatten(1), dim=1)
    
    def _calculate_recall(self, scores: torch.Tensor, mask: torch.Tensor, prefix: str):
        rankings = scores.argsort(dim=1, descending=True)
        matched_positions = mask.gather(1, rankings).float()
        positions = torch.argmax(matched_positions, dim=1)
        #positions = (rankings == target_indices[:, None]).nonzero(as_tuple=False)[:, 1]
        
        results = {}
        results[f"{prefix}R1"] = (positions < 1).float().mean().item()
        results[f"{prefix}R5"] = (positions < 5).float().mean().item()
        results[f"{prefix}R10"] = (positions < 10).float().mean().item()
        results[f"{prefix}_mrr"] = (1.0 / (positions.float() + 1.0)).mean().item()
        return results

    @torch.no_grad()
    def global_retrieval(self, dataloader, batch_mapper) -> dict:
        self.image_model.eval()
        self.text_model.eval()
        all_img, all_txt = [], []
        global_img_idx, global_txt_idx = 0, 0
        match_coordinates = []

        for batch in dataloader:
            img_batch, txt_batch = batch_mapper(batch)
            img_emb = self._encode_img(img_batch)
            all_img.append(img_emb.cpu())

            text_payload_batch = []
            for item_captions in txt_batch:
                if isinstance(item_captions[0], str) or isinstance(item_captions[0][0], dict):
                    # Single element, either (str, [dict]) or ([dict], None)
                    item_captions = [item_captions] 

                num_captions = len(item_captions)
                for _ in range(num_captions):
                    match_coordinates.append((global_img_idx, global_txt_idx))
                    global_txt_idx += 1
                global_img_idx += 1

                for c in item_captions:
                    text_payload_batch.append((c[0], c[1]))
            txt_emb = self._encode_txt(text_payload_batch)
            all_txt.append(txt_emb.cpu())

        similarity = (torch.cat(all_img, dim=0) @ torch.cat(all_txt, dim=0).T).abs()
        n_imgs, m_txts = similarity.shape

        mask = torch.zeros((n_imgs, m_txts), dtype=torch.bool, device=similarity.device)
        for i, j in match_coordinates:
            mask[i, j] = True

        metrics = {}

        # --- 1. Directional Retrieval Metrics (i2t and t2i) ---
        metrics.update(self._calculate_recall(similarity, mask, "i2t"))
        metrics.update(self._calculate_recall(similarity.T, mask.T, "t2i"))

        # --- 2. Advanced Geometric & Structural Latent Metrics ---
        avg_positive = similarity[mask].mean().item()
        avg_negative = similarity[~mask].mean().item()

        # --- 3. Global Embedding Margin ---
        metrics["gamma"] = avg_positive - avg_negative

        # --- 4. Directional Hard-Negative Margins ---
        mask_matrix = similarity.clone()
        mask_matrix[mask] = float("-inf")
        
        max_neg_i2t, _ = mask_matrix.max(dim=1)
        max_neg_t2i, _ = mask_matrix.max(dim=0)

        pos_matrix = similarity.clone()
        pos_matrix[~mask] = 0.0
        avg_pos_per_img = pos_matrix.sum(dim=1) / mask.sum(dim=1).float()
        avg_pos_per_txt = pos_matrix.sum(dim=0) / mask.sum(dim=0).float()
        
        metrics["i2t_hnm"] = (avg_pos_per_img - max_neg_i2t).mean().item()
        metrics["t2i_hnm"] = (avg_pos_per_txt - max_neg_t2i).mean().item()

        # --- 5. Spatial Alignment Asymmetry ---
        metrics["delta"] = abs(metrics["i2tR1"] - metrics["t2iR1"])
            
        return metrics
    
    @torch.no_grad()
    def evaluate_text_choice(self, dataloader) -> float:
        self.image_model.eval()
        self.text_model.eval()
        correct = total = 0
        for batch in dataloader:
            img_emb = self._encode_img(batch["image"])
            pos_txt_emb = self._encode_txt(batch["pos_caption"])
            neg_txt_emb = self._encode_txt(batch["neg_caption"])
            
            pos_sim = torch.sum(img_emb * pos_txt_emb, dim=1)
            neg_sim = torch.sum(img_emb * neg_txt_emb, dim=1)
            
            correct += (pos_sim > neg_sim).sum().item()
            total += img_emb.size(0)
        return {"text_choice": correct / total}


    @torch.no_grad()
    def evaluate_image_choice(self, dataloader) -> float:
        correct = total = 0
        for batch in dataloader:
            txt_emb = self._encode_txt(batch["caption"])
            pos_img_emb = self._encode_img(batch["pos_image"])
            neg_img_emb = self._encode_img(batch["neg_image"])
            
            pos_sim = torch.sum(txt_emb * pos_img_emb, dim=1).abs()
            neg_sim = torch.sum(txt_emb * neg_img_emb, dim=1).abs()
            
            correct += (pos_sim > neg_sim).sum().item()
            total += txt_emb.size(0)
        return {"acc": correct / total}
    
    @torch.no_grad()
    def evaluate_sugarcrepe_pp(self, dataloader: torch.utils.data.DataLoader) -> float:
        correct = total = 0
        for batch in dataloader:
            img_emb = self._encode_img(batch["image"])
            pos1_emb = self._encode_txt(batch["pos_caption1"])
            pos2_emb = self._encode_txt(batch["pos_caption2"])
            neg_emb = self._encode_txt(batch["neg_caption"])
            
            sim_pos1 = torch.sum(img_emb * pos1_emb, dim=1)
            sim_pos2 = torch.sum(img_emb * pos2_emb, dim=1)
            sim_neg  = torch.sum(img_emb * neg_emb, dim=1)
            
            match = (sim_pos1 > sim_neg) & (sim_pos2 > sim_neg)
            correct += match.sum().item()
            total += img_emb.size(0)
            
        return {"acc": correct / total}

    @torch.no_grad()
    def evaluate_winoground(self, dataloader) -> dict:
        text_corr = img_corr = group_corr = total = 0
        for batch in dataloader:
            i0, c0 = self._encode_img(batch["image_0"]), self._encode_txt(batch["caption_0"])
            i1, c1 = self._encode_img(batch["image_1"]), self._encode_txt(batch["caption_1"])
            
            s_i0_c0 = torch.sum(i0 * c0, dim=1)
            s_i0_c1 = torch.sum(i0 * c1, dim=1)
            s_i1_c0 = torch.sum(i1 * c0, dim=1)
            s_i1_c1 = torch.sum(i1 * c1, dim=1)

            t_match = (s_i0_c0 > s_i0_c1) & (s_i1_c1 > s_i1_c0)
            i_match = (s_i0_c0 > s_i1_c0) & (s_i1_c1 > s_i0_c1)
            g_match = t_match & i_match

            text_corr += t_match.sum().item()
            img_corr += i_match.sum().item()
            group_corr += g_match.sum().item()
            total += i0.size(0)
        
        return {"txt_score": text_corr/total, "img_score": img_corr/total, "grp_score": group_corr/total}
    
    @torch.no_grad()
    def evaluate_swap(self, dataloader) -> float:
        self.image_model.eval()
        self.text_model.eval()
        correct = total = 0
        for batch in dataloader:
            img_emb = self._encode_img(batch["image"])
            pos_txt_emb = self._encode_txt(batch["pos_caption"])
            neg_txt_emb = self._encode_txt(batch["neg_caption"])
            
            pos_sim = torch.sum(img_emb * pos_txt_emb, dim=1).abs()
            neg_sim = torch.sum(img_emb * neg_txt_emb, dim=1).abs()
            
            correct += (pos_sim > neg_sim).sum().item()
            total += img_emb.size(0)
        return {"swap_acc": correct / total}
    
    @torch.no_grad()
    def compositional_diagnostic(self, dataloader):
        """
        Calculates the explicit similarity distributions between true and foil pairs
        to detect representation collapse and semantic smearing.
        """
        self.image_model.eval()
        self.text_model.eval()

        sum_pos_sim = 0.0
        sum_neg_sim = 0.0
        sum_absolute_gap = 0.0
        total_samples = 0

        for batch in dataloader:
            # txt_emb = self._encode_txt(batch["caption"])
            # pos_img_emb = self._encode_img(batch["pos_image"])
            # neg_img_emb = self._encode_img(batch["neg_image"])
            txt_emb = self._encode_txt(batch["image"])
            pos_img_emb = self._encode_img(batch["pos_caption"])
            neg_img_emb = self._encode_img(batch["neg_caption"])
            
            # Replicating your model's native similarity metric calculation
            pos_sim = torch.sum(txt_emb * pos_img_emb, dim=1).abs()
            neg_sim = torch.sum(txt_emb * neg_img_emb, dim=1).abs()
            
            # Compute the absolute distance between the positive and negative scores per sample
            batch_gap = (pos_sim - neg_sim).abs()

            sum_pos_sim += pos_sim.sum().item()
            sum_neg_sim += neg_sim.sum().item()
            sum_absolute_gap += batch_gap.sum().item()
            total_samples += txt_emb.size(0)

        if total_samples == 0:
            return {}

        return {
            "diag_mean_pos_overlap": sum_pos_sim / total_samples,
            "diag_mean_neg_overlap": sum_neg_sim / total_samples,
            "diag_collapse_gap": sum_absolute_gap / total_samples
        }