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
from modules.utils.general import get_device, load_pkl
from modules.data_pipeline.datasets import SwapDataset, swap_collate_fn

def log_phase(name: str):
    print(f"\n[{name.upper()}] " + "—" * (60 - len(name)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
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

    log_phase("Compiling Evaluation Graph Structures")
    df_eval = load_pkl(config["splits"]['swap']["text_path"])

    compile_kwargs = {}
    if config["model_type"] == "vqc":
        compile_kwargs["curry"] = config["compiler"].get("curry", False)
    compiled_eval = ansatz.compile_dataset(df_eval, **compile_kwargs)

    log_phase("Restoring Dynamic Parameter Spaces")

    df_train = load_pkl(config["splits"]["train"]["text_path"])
    df_val = load_pkl(config["splits"]["val"]["text_path"])
    compiled_train = ansatz.compile_dataset(df_train, **compile_kwargs)
    compiled_val = ansatz.compile_dataset(df_val, **compile_kwargs)

    if hasattr(text_model, "from_symbols"):
        sym_cols = [c for c in compiled_train.columns if c.endswith("_symbols")]
        txt_stream = []
        for col in sym_cols:
            txt_stream += compiled_train[col].tolist() + compiled_val[col].tolist()
        sym_kwargs = {"id_init": True} if config["model_type"] == "vqc" else {}
        text_model.from_symbols(txt_stream, **sym_kwargs)

    if hasattr(image_model, "fit_image_pca"):
        train_embeddings = torch.load(config["splits"]["train"]["img_path"])
        raw_tensor_stack = torch.stack(list(train_embeddings.values())).to(DEV)
        image_model.fit_image_pca(raw_tensor_stack)

    log_phase("Loading Model Checkpoint Weights")
    checkpoint = torch.load(checkpoint_path, map_location=DEV)
    img_params = sum(p.numel() for p in checkpoint["image"].values() if hasattr(p, "numel"))
    txt_params = sum(p.numel() for p in checkpoint["text"].values() if hasattr(p, "numel"))
    print(f"    [Checkpoint Info] -> Image Params: {img_params:,} | Text Params: {txt_params:,}")

    image_model.load_state_dict(checkpoint["image"])
    text_model.load_state_dict(checkpoint["text"])

    saved_epoch = checkpoint.get("epoch", "N/A")
    saved_loss = checkpoint.get("train_loss", "N/A")

    print(f"    [Checkpoint Info] -> Recovered from Epoch: {saved_epoch} | Historical Loss: {saved_loss}")

    image_model.eval()
    text_model.eval()

    log_phase("Constructing Data Engine Context")
    if config['vision']['use_clip']:
        img_transform = None
    else:
        img_transform = v2.Compose([v2.ToImage(),               
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Resize((64, 64))])
        
    eval_loader = DataLoader(
        SwapDataset(compiled_eval, config["splits"]['swap']["img_path"], image_transform=img_transform),
        batch_size=config["batch_size"], 
        collate_fn=swap_collate_fn, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    log_phase("Executing Metrics Benchmark Suite")
    evaluator = MMEvaluator(image_model, text_model, DEV)

    metrics = {}

    with torch.no_grad():
        start_task = time.time()
        task_metrics = evaluator.evaluate_swap(eval_loader)
        metrics.update(task_metrics)
        elapsed = time.time() - start_task
        print(f"    Completed in {elapsed:.2f}s")

    log_phase("Final Benchmark Scoreboard")
    print(f" {'Metric Key':<35} | {'Value / Score':<15}")
    print(" " + "—" * 53)
    for key, val in metrics.items():
        print(f"  {key:<34} | {val:.6f}")
    print(" " + "—" * 53 + "\n")