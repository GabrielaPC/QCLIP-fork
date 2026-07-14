import argparse, logging, os, time, yaml
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

# Local framework imports
from factory import build_dataset, build_experiment
from modules.models.fusion.engine import MMEvaluator
from modules.utils.general import get_device, load_pkl, store_pkl
from modules.data_pipeline.datasets import SwapDataset, swap_collate_fn
from modules.utils.analysis import tn_metadata, analyse_einsum
from modules.utils.general import gen_id

def log_phase(name: str):
    print(f"\n[{name.upper()}] " + "—" * (60 - len(name)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument("-cp", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    args = parser.parse_args()

    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    DEV = get_device()
    checkpoint_path = Path(args.checkpoint)

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV}")
    print(f" Loading Weights : {checkpoint_path.name}")
    print(f" Base Config     : {args.config}")

    log_phase("Instantiating Architecture")
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    DatasetClass, collate_fn, eval_mapper = build_dataset(config)

    compile_kwargs = {}
    if config["model_type"] == "vqc":
        compile_kwargs["curry"] = config["text"].get("curry", False)

    log_phase("Restoring Dynamic Parameter Spaces")

    df_train = load_pkl(config["splits"]["train"]["text_path"])
    df_val = load_pkl(config["splits"]["val"]["text_path"])
    compiled_train = ansatz.compile_dataset(df_train, **compile_kwargs)
    compiled_val = ansatz.compile_dataset(df_val, **compile_kwargs)

    if hasattr(text_model, "from_symbols"):
        sym_cols = [c for c in compiled_train.columns if c.endswith("_symbols")]
        symbol_arr = []
        for col in sym_cols:
            symbol_arr += compiled_train[col].tolist() + compiled_val[col].tolist()
        sym_kwargs = {"id_init": True} if config["model_type"] == "vqc" else {}
        text_model.from_symbols(symbol_arr, **sym_kwargs)

        einsum_cols = [c for c in compiled_train.columns if c.endswith("_einsum")]
        einsum_arr = []
        for col in einsum_cols:
            einsum_arr += compiled_train[col].tolist() + compiled_val[col].tolist()

        tn_arr = list(zip(einsum_arr, symbol_arr))
        metrics = tn_metadata(tn_arr)
        print(f"Circuit Metrics for Text Model:")
        print(f"(Max) Qubits: {metrics['max'][0]:.4f} | Gates: {metrics['max'][1]:.4f} | Depth: {metrics['max'][2]:.4f} | Rank: {metrics['max'][3]:.4f}")
        print(f"(Avg) Qubits: {metrics['avg'][0]:.4f} | Gates: {metrics['avg'][1]:.4f} | Depth: {metrics['avg'][2]:.4f} | Rank: {metrics['avg'][3]:.4f}")

    if config['vision']['method'] == 'pca':
        train_embeddings = torch.load(config["splits"]["train"]["img_path"])
        raw_tensor_stack = torch.stack(list(train_embeddings.values())).to(DEV)
        image_model.fit_image_pca(raw_tensor_stack)
    if config['vision']['method'] in ['mlp', 'pca']:
        metrics = analyse_einsum(image_model.einsum_expr.replace('b', ''), image_model.gate_arr)
        print(f"Circuit Metrics for Image Model:")
        print(f"Qubits: {metrics[0]:.4f} | Gates: {metrics[1]:.4f} | Depth: {metrics[2]:.4f} | Rank: {metrics[3]:.4f}")

    log_phase("Loading Model Checkpoint Weights")
    checkpoint = torch.load(checkpoint_path, map_location=DEV)
    text_params = checkpoint["text"]
    image_params = checkpoint["image"]
    n_img_params = sum(p.numel() for p in image_params.values() if hasattr(p, "numel"))
    n_txt_params = sum(p.numel() for p in text_params.values() if hasattr(p, "numel"))
    print(f"    [Checkpoint Info] -> Image Params: {n_img_params:,} | Text Params: {n_txt_params:,}")

    def upgrade_checkpoint(old_state_dict):
        new_state_dict = {}
        for key, value in old_state_dict.items():
            if key == "params":
                for i in range(value.size(0)):
                    new_state_dict[f"params.{i}"] = value[i:i+1]
            else:
                new_state_dict[key] = value
        return new_state_dict

    if "params" in text_params and not any("params." in k for k in text_params):
        text_params = upgrade_checkpoint(text_params)
    if "params" in image_params and not any("params." in k for k in image_params):
        image_params = upgrade_checkpoint(image_params)

    text_model.load_state_dict(text_params, strict=False)
    image_model.load_state_dict(image_params)

    saved_epoch = checkpoint.get("epoch", "N/A")
    saved_loss = checkpoint.get("train_loss", "N/A")

    print(f" [Checkpoint Info] -> Recovered from Epoch: {saved_epoch} | Historical Loss: {saved_loss}")

    image_model.eval()
    text_model.eval()

    if config['vision']['method'] == 'amp':
        img_transform = None
    else:
        img_transform = v2.Compose([v2.ToImage(),               
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Resize((64, 64))])        

    log_phase("Compiling Evaluation Graph Structures")
    test_sets = config["splits"]["test"]
    evaluator = MMEvaluator(image_model, text_model, DEV)

    all_results = {}
    for split_name, split_info in test_sets.items():
        if split_name == 'swap':
            DatasetClass = SwapDataset
            collate_fn = swap_collate_fn

        df_eval = load_pkl(split_info["text_path"])
        compiled_eval = ansatz.compile_dataset(df_eval, **compile_kwargs)
        eval_loader = DataLoader(
            DatasetClass(compiled_eval, split_info["img_path"], image_transform=img_transform),
            batch_size=config["batch_size"], 
            collate_fn=collate_fn, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        print(f" -> DataLoader ready for '{split_name}': {len(eval_loader)} steps")

        eval_fn = getattr(evaluator, split_info['task'], None)

        with torch.no_grad():
            start_task = time.time()
            task_metrics, task_deta = eval_fn(eval_loader)
            elapsed = time.time() - start_task
            print(f" -> Evaluated '{split_name}' task in {elapsed:.2f}s")
            for key, val in task_metrics.items():
                all_results[f"{split_name}_{key}"] = val
            run_name = gen_id(config)
            save_filename = f"results/{config['dataset']['name']}/{run_name}_{split_name}_margins.pkl"
            os.makedirs(os.path.dirname(save_filename), exist_ok=True)
            store_pkl(task_deta, save_filename)

    log_phase("Final Benchmark Scoreboard")
    print(f"{'Split & Metric Key':<35} | {'Value / Score':<15}")
    print("—" * 53)
    for key, val in all_results.items():
        print(f"{key:<35} | {val:.6f}")
    print("—" * 53 + "\n")