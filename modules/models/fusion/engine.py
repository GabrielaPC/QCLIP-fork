import torch, yaml, mlflow, time, socket, math
from collections import defaultdict
from typing import Callable
import torch.nn.functional as F
from modules.utils.general import gen_id, log_phase
from pathlib import Path
from tqdm import tqdm

mscoco_mapper = lambda batch: (batch["image"], batch["caption"])
aro_mapper = lambda batch: (batch["image"], batch["pos_caption"])
svo_mapper = lambda batch: (batch["pos_image"], batch["caption"])


class RunManager:
    def __init__(self, config, trainer, evaluator, device, seed, scheduler=None):
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self.device = device
        self.seed = seed
        self.scheduler = scheduler

        self.dataset_name = config['dataset']['name']
        self.model_type = config['model_type']
        self.run_name = gen_id(config)

        self.checkpoint_dir = Path(f"./checkpoints/{self.dataset_name}/{self.model_type}/{self.run_name}")
        self.checkpoint_path = self.checkpoint_dir / f"last.pt"
        self.best_checkpoint_path = self.checkpoint_dir / f"best.pt"

        self.eval_tasks = config['dataset']['val'].get('diagnostics', ['global_retrieval'])

    def _setup_environment(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_dir / f"config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        mlf_db_path = Path.cwd() / f"mlf_dbs/{self.dataset_name}.db"
        mlf_db_path.parent.mkdir(parents=True, exist_ok=True)

        mlflow.pytorch.autolog(log_models=False)
        mlflow.set_tracking_uri(f"sqlite:///{mlf_db_path}")
        mlflow.set_experiment(self.dataset_name+'_train')

    def _save_checkpoint(self, epoch, loss, metrics, is_best=False):
        payload = {
            "image": self.trainer.image_model.state_dict(),
            "text": self.trainer.text_model.state_dict(),
            "epoch": epoch,
            "train_loss": loss,
            "val_metrics": metrics
        }
        torch.save(payload, self.checkpoint_path)
        if is_best:
            torch.save(payload, self.best_checkpoint_path)

    def fit(self, train_loader, val_loader, eval_mapper):
        self._setup_environment()
        hostname = socket.gethostname()
        log_phase(f"Model #{self.run_name}: optimization started on \"{hostname}\"...")
        
        best_metric_value = float("-inf")
        target_metric = "i2tR1" if 'global_retrieval' in self.eval_tasks else 'hard_neg_acc'

        patience = self.config.get('patience', 15)
        min_delta = self.config.get('min_delta', 1e-3)
        patience_counter = 0
        stopped_epoch = 0

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_params({
                "epochs": self.config['epochs'],
                "batch_size": self.config['batch_size'],
                "learning_rate_quantum": self.config['qlr'],
                "learning_rate_classical": self.config['clr'],
                "embedding_qubits": self.config.get('embedding_qubits', 0),
                "temperature_parameter": self.trainer.loss_fn.temperature,
                "device_target": str(self.device),
                "seed": self.seed,
                "text_tower": type(self.trainer.text_model).__name__,
                "image_tower": type(self.trainer.image_model).__name__,
                "execution_host": hostname,
                "model_path": str(self.checkpoint_path),
                "early_stopping_patience": patience,
                "early_stopping_min_delta": min_delta,
            })

            epoch_pbar = tqdm(range(self.config['epochs']), desc="Training Pipeline", unit="epoch")
            current_lr = self.config.get('qlr', 1e-3)
            
            for epoch in epoch_pbar:
                start_time = time.time()
                loss = self.trainer.train_epoch(train_loader, eval_mapper)
                elapsed_time = time.time() - start_time

                grad_norm_dict = self.trainer.grad_norm_by_type()

                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

                metrics = self.evaluator.eval_set(
                    dataloader=val_loader,
                    tasks=self.eval_tasks,
                    eval_mapper=eval_mapper
                )

                mlflow.log_metrics(metrics, step=epoch)
                for k, v in grad_norm_dict.items():
                    mlflow.log_metric(f"gradient_norm_{k}", v, step=epoch)
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                tqdm.write(f"\nEpoch {epoch:02d} | Loss: {loss:.4f} | Gradient Norms: {', '.join([f'{k}: {v:.4f}' for k, v in grad_norm_dict.items()])} | {metrics_str} | Time: {elapsed_time:.1f}s")

                current_metric = metrics.get(target_metric, 0.0)
                if current_metric > (best_metric_value + min_delta):
                    best_metric_value = current_metric
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False

                self._save_checkpoint(epoch, loss, metrics, is_best=is_best)

                if patience_counter >= patience:
                    stopped_epoch = epoch
                    tqdm.write(
                        f"\n[EARLY STOPPING TRIGGERED] Metric '{target_metric}' failed to improve by > {min_delta} "
                        f"for {patience} consecutive epochs. Best Score: {best_metric_value:.4f} (at epoch {epoch - patience})."
                    )
                    mlflow.log_metric("early_stopped_epoch", stopped_epoch)
                    break
                
        log_phase("Experiment Run Concluded")

class ContrastiveTrainer:
    def __init__(self, image_model, text_model, optimizer, loss_fn, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    @torch.no_grad()
    def grad_norm(self):
        grad_norm = 0.0

        for _, param in list(self.image_model.named_parameters()) + list(self.text_model.named_parameters()):
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5

    @torch.no_grad()
    def grad_norm_by_type(self) -> dict:
        verb_grad_sq = 0.0
        noun_grad_sq = 0.0
        other_grad_sq = 0.0

        sym2param = getattr(self.text_model, 'sym2param', {})
        param2sym = {idx: sym for sym, idx in sym2param.items()}

        for name, param in self.text_model.named_parameters():
            if param.grad is None:
                continue

            g_norm_sq = param.grad.data.norm(2).item() ** 2

            if "params" in name:
                try:
                    idx = int(name.split(".")[-1])
                    sym_name = param2sym.get(idx, "")
                except ValueError:
                    sym_name = ""
            else:
                sym_name = name

            # print(sym_name.split('__')[1].split('_')[0].split('@'))
            cmplx_type = sym_name.split('__')[1].split('_')[0].split('@')
            op_arity = len(cmplx_type)
            cmplx_type = ''.join(cmplx_type)

            if cmplx_type == 'n':
                noun_grad_sq += g_norm_sq
            elif op_arity == 3:
                verb_grad_sq += g_norm_sq
            else:
                other_grad_sq += g_norm_sq

        return {
            "verb_grad_norm": verb_grad_sq ** 0.5,
            "noun_grad_norm": noun_grad_sq ** 0.5,
            "other_grad_norm": other_grad_sq ** 0.5,
            "total_text_grad_norm": (verb_grad_sq + noun_grad_sq) ** 0.5
        }

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

            # torch.nn.utils.clip_grad_norm_(self.text_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item() * images.shape[0]
        
        return epoch_loss / len(dataloader.dataset)

def adaptive_optimizer(text_model, image_model, base_qlr: float = 1e-3, base_clr: float = 1e-5, 
                       arity_factor: float = 1.5, weight_decay: float = 0.01) -> torch.optim.AdamW:
    
    sym2param = getattr(text_model, 'sym2param', {})
    param2sym = {idx: sym for sym, idx in sym2param.items()}
    arity_param_groups = defaultdict(list)

    for name, param in text_model.named_parameters():
        if not param.requires_grad:
            continue

        if "params" in name:
            try:
                idx = int(name.split(".")[-1])
                sym_name = param2sym.get(idx, "")
            except ValueError:
                sym_name = ""
        else:
            sym_name = name

        cmplx_type = sym_name.split('__')[1].split('_')[0].split('@')
        op_arity = max(1, len(cmplx_type))
        arity_param_groups[op_arity].append(param)

    param_groups = []
    
    for arity in sorted(arity_param_groups.keys()):
        params = arity_param_groups[arity]
        group_lr = base_qlr * (arity_factor ** (arity - 1))
        
        param_groups.append({
            'params': params,
            'lr': group_lr,
            'name': f'q_text_arity_{arity}_lr_{group_lr:.1e}'
        })
        print(f"[Optimizer Config] Arity {arity} | {len(params)} params | LR: {group_lr:.6f}")

    # Add classical image tower group
    param_groups.append({
        'params': image_model.parameters(),
        'lr': base_clr,
        'name': 'image_tower'
    })
    print("[Optimizer Config] Classical Image Tower | {} params | LR: {:.6f}".format(len(list(image_model.parameters())), base_clr))
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

class MMEvaluator:
    def __init__(self, image_model, text_model, device):
        self.image_model = image_model.to(device)
        self.text_model = text_model.to(device)
        self.device = device

    @torch.no_grad()
    def _encode_txt(self, texts):
        return self.text_model(texts).flatten(1).contiguous()
    
    @torch.no_grad()
    def _encode_img(self, images):
        return self.image_model(images.to(self.device)).flatten(1).contiguous()
    
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

        similarity = (torch.cat(all_img, dim=0) @ torch.cat(all_txt, dim=0).conj().T).abs()
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
    def hard_neg_eval(self, dataloader, choice = 'text') -> dict:
        self.image_model.eval()
        self.text_model.eval()
        correct_final = total = 0

        pos_arr, neg_arr = [], []
        correct_arr, margin_arr = [], []
        for batch in dataloader:
            if choice == 'text':
                m1_emb = self._encode_img(batch["image"])
                pos_m2_emb = self._encode_txt(batch["pos_caption"])
                neg_m2_emb = self._encode_txt(batch["neg_caption"])
            if choice == 'image':
                m1_emb = self._encode_txt(batch["caption"])
                pos_m2_emb = self._encode_img(batch["pos_image"])
                neg_m2_emb = self._encode_img(batch["neg_image"])

            pos_sim = torch.sum(m1_emb.conj() * pos_m2_emb, dim=1).abs()
            neg_sim = torch.sum(m1_emb.conj() * neg_m2_emb, dim=1).abs()
            pos_arr.extend(pos_sim.cpu().numpy())
            neg_arr.extend(neg_sim.cpu().numpy())
            
            margin = pos_sim - neg_sim
            correct = (margin > 0).float()
            correct_final += correct.sum().item()
            margin_arr.extend(margin.cpu().numpy())
            correct_arr.extend(correct.cpu().numpy())

            total += m1_emb.size(0)
        return {"hard_neg_acc": correct_final / total}, {"pos_scores": pos_arr, "neg_scores": neg_arr, "margins": margin_arr, "correct": correct_arr}
            
    def evaluate_text_choice(self, dataloader) -> dict:
        return self.hard_neg_eval(dataloader, choice='text')

    def evaluate_image_choice(self, dataloader) -> dict:
        return self.hard_neg_eval(dataloader, choice='image')

    @torch.no_grad()
    def evaluate_sugarcrepe_pp(self, dataloader: torch.utils.data.DataLoader) -> float:
        correct = total = 0
        for batch in dataloader:
            img_emb = self._encode_img(batch["image"])
            pos1_emb = self._encode_txt(batch["pos_caption1"])
            pos2_emb = self._encode_txt(batch["pos_caption2"])
            neg_emb = self._encode_txt(batch["neg_caption"])
            
            sim_pos1 = torch.sum(img_emb.conj() * pos1_emb, dim=1).abs()
            sim_pos2 = torch.sum(img_emb.conj() * pos2_emb, dim=1).abs()
            sim_neg  = torch.sum(img_emb.conj() * neg_emb, dim=1).abs()
            
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
            
            s_i0_c0 = torch.sum(i0.conj() * c0, dim=1).abs()
            s_i0_c1 = torch.sum(i0.conj() * c1, dim=1).abs()
            s_i1_c0 = torch.sum(i1.conj() * c0, dim=1).abs()
            s_i1_c1 = torch.sum(i1.conj() * c1, dim=1).abs()

            t_match = (s_i0_c0 > s_i0_c1) & (s_i1_c1 > s_i1_c0)
            i_match = (s_i0_c0 > s_i1_c0) & (s_i1_c1 > s_i0_c1)
            g_match = t_match & i_match

            text_corr += t_match.sum().item()
            img_corr += i_match.sum().item()
            group_corr += g_match.sum().item()
            total += i0.size(0)
        
        return {"txt_score": text_corr/total, "img_score": img_corr/total, "grp_score": group_corr/total}

    @torch.no_grad()
    def cptp_diagnostic(self, dataloader, choice="text") -> dict:
        self.image_model.eval()
        self.text_model.eval()

        sum_m1_purity, sum_m1_entropy, sum_m1_eff_rank = 0.0, 0.0, 0.0
        sum_m2_purity, sum_m2_entropy, sum_m2_eff_rank = 0.0, 0.0, 0.0
        total_samples = 0
        metrics = {}

        def _to_density_matrix(emb: torch.Tensor) -> torch.Tensor:
            B = emb.size(0)
            if emb.dim() == 2:
                D = int(math.sqrt(emb.size(1)))
                return emb.view(B, D, D).to(torch.complex64)
            return emb.to(torch.complex64)

        def _compute_state_stats(rho: torch.Tensor):
            # 1. Purity P = Tr(rho^2) computed natively on GPU
            purity = torch.real(torch.einsum('bmn,bnm->b', rho, rho))

            # 2. CPU Offloading for Hermitian Eigenvalues to avoid cuSOLVER/CUDA driver crashes
            rho_cpu = rho.detach().cpu()
            evals = torch.linalg.eigvalsh(rho_cpu).clamp(min=1e-8)

            # 3. von Neumann Entropy S(rho) = -Tr(rho ln rho)
            entropy = -(evals * torch.log(evals)).sum(dim=-1)

            # 4. Effective Rank / Participation Ratio R_eff = 1 / Tr(rho^2)
            eff_rank = 1.0 / (evals ** 2).sum(dim=-1)

            return purity.sum().item(), entropy.sum().item(), eff_rank.sum().item()

        for batch in dataloader:
            if choice == "text":
                m1_emb = self._encode_img(batch["image"])
                pos_m2_emb = self._encode_txt(batch["pos_caption"])
                neg_m2_emb = self._encode_txt(batch["neg_caption"])
            elif choice == "image":
                m1_emb = self._encode_txt(batch["caption"])
                pos_m2_emb = self._encode_img(batch["pos_image"])
                neg_m2_emb = self._encode_img(batch["neg_image"])

            B = m1_emb.size(0)
            rho_m1 = _to_density_matrix(m1_emb)
            rho_m2_pos = _to_density_matrix(pos_m2_emb)
            rho_m2_neg = _to_density_matrix(neg_m2_emb)

            # Modality 1 statistics
            m1_pur, m1_ent, m1_rank = _compute_state_stats(rho_m1)
            sum_m1_purity += m1_pur
            sum_m1_entropy += m1_ent
            sum_m1_eff_rank += m1_rank

            # Modality 2 statistics (averaged across positive and negative pairs)
            m2_pos_pur, m2_pos_ent, m2_pos_rank = _compute_state_stats(rho_m2_pos)
            m2_neg_pur, m2_neg_ent, m2_neg_rank = _compute_state_stats(rho_m2_neg)

            sum_m2_purity += 0.5 * (m2_pos_pur + m2_neg_pur)
            sum_m2_entropy += 0.5 * (m2_pos_ent + m2_neg_ent)
            sum_m2_eff_rank += 0.5 * (m2_pos_rank + m2_neg_rank)

            total_samples += B

        if total_samples == 0:
            return {}

        # Map modality identifiers based on query choice
        m1_name = "img" if choice == "text" else "txt"
        m2_name = "txt" if choice == "text" else "img"

        mean_m1_pur = sum_m1_purity / total_samples
        mean_m2_pur = sum_m2_purity / total_samples

        # 1. Purities & Purity Imbalance Ratio
        metrics[f"diag_mean_{m1_name}_purity"] = mean_m1_pur
        metrics[f"diag_mean_{m2_name}_purity"] = mean_m2_pur
        metrics["diag_purity_imbalance"] = abs(mean_m1_pur - mean_m2_pur)

        # 2. von Neumann Entropies
        metrics[f"diag_mean_{m1_name}_entropy"] = sum_m1_entropy / total_samples
        metrics[f"diag_mean_{m2_name}_entropy"] = sum_m2_entropy / total_samples

        # 3. Effective Ranks / Participation Ratios
        metrics[f"diag_mean_{m1_name}_eff_rank"] = sum_m1_eff_rank / total_samples
        metrics[f"diag_mean_{m2_name}_eff_rank"] = sum_m2_eff_rank / total_samples

        # Legacy backward-compatibility key
        metrics["diag_mean_text_purity"] = metrics["diag_mean_txt_purity"]

        return metrics
    
    @torch.no_grad()
    def diagnostic(self, dataloader, choice="text") -> dict:
        self.image_model.eval()
        self.text_model.eval()

        sum_pos_sim = 0.0
        sum_neg_sim = 0.0
        sum_absolute_gap = 0.0
        total_samples = 0

        metrics = {}

        for batch in dataloader:
            if choice == "text":
                m1_emb = self._encode_img(batch["image"])
                pos_m2_emb = self._encode_txt(batch["pos_caption"])
                neg_m2_emb = self._encode_txt(batch["neg_caption"])
                text_tensors = [pos_m2_emb, neg_m2_emb]
            elif choice == "image":
                m1_emb = self._encode_txt(batch["caption"])
                pos_m2_emb = self._encode_img(batch["pos_image"])
                neg_m2_emb = self._encode_img(batch["neg_image"])
                text_tensors = [m1_emb]
            
            # Replicating your model's native similarity metric calculation
            pos_sim = torch.sum(m1_emb.conj() * pos_m2_emb, dim=1).abs()
            neg_sim = torch.sum(m1_emb.conj() * neg_m2_emb, dim=1).abs()
            
            # Compute the absolute distance between the positive and negative scores per sample
            batch_gap = (pos_sim - neg_sim).abs()

            sum_pos_sim += pos_sim.sum().item()
            sum_neg_sim += neg_sim.sum().item()
            sum_absolute_gap += batch_gap.sum().item()
            total_samples += m1_emb.size(0)

        if total_samples == 0:
            return {}
        
        metrics["diag_mean_pos_overlap"] = sum_pos_sim / total_samples
        metrics["diag_mean_neg_overlap"] = sum_neg_sim / total_samples
        metrics["diag_collapse_gap"] = sum_absolute_gap / total_samples

        return metrics
    
    def diagnostic_text_choice(self, dataloader) -> dict:
        return self.diagnostic(dataloader, choice="text")
    
    def diagnostic_image_choice(self, dataloader) -> dict:
        return self.diagnostic(dataloader, choice="image")

    def cptp_diagnostic_text_choice(self, dataloader) -> dict:
        return self.cptp_diagnostic(dataloader, choice="text")

    def cptp_diagnostic_image_choice(self, dataloader) -> dict:
        return self.cptp_diagnostic(dataloader, choice="image")
    
    @torch.no_grad()
    def eval_set(self, dataloader, tasks, eval_mapper) -> dict:
        self.image_model.eval()
        self.text_model.eval()
        metrics = {}
        for task_name in tasks:
            eval_fn = getattr(self, task_name, None)
            if eval_fn is None:
                print(f"Warning: Evaluation method '{task_name}' not found on MMEvaluator. Skipping.")
                continue

            if task_name == "global_retrieval":
                task_metrics = eval_fn(dataloader, eval_mapper)
            else:
                out = eval_fn(dataloader)
                task_metrics = out[0] if isinstance(out, tuple) else out
                
            metrics.update(task_metrics)
            
        return metrics