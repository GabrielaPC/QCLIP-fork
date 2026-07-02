import torch, mlflow, time, yaml, argparse, os, logging, socket, platform
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pathlib import Path
from datetime import datetime

from factory import build_experiment, build_dataset
from modules.utils.general import load_pkl, get_device, set_seed, gen_id, log_phase
from modules.models.fusion.engine import ContrastiveTrainer, MMEvaluator

from tqdm import tqdm

# uv run python train.py --config configs/tensor_network.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Environment setup
    log_phase("Environment Initialized")
    DEV = get_device()
    SEED = int.from_bytes(os.urandom(4))
    set_seed(SEED)

    ROOT_PATH = Path.cwd()
    DATASET = config['dataset']['name']
    BATCH_SIZE = config['batch_size']
    print(f" Target Device : {DEV}")
    print(f" Random Seed   : {SEED}")
    print(f" Run Identifier: {DATASET}")

    # Extract experiment components
    log_phase("Setting Up Experiment Components")
    ansatz, image_model, text_model, loss_fn = build_experiment(config, DEV)
    DatasetClass, collate_fn, eval_mapper = build_dataset(config)
    print(" Component structures built successfully.")
    
    # Load and compile datasets
    log_phase("Compiling Symbolic Datasets")
    df_train = load_pkl(config['splits']['train']['text_path'])
    df_val = load_pkl(config['splits']['val']['text_path'])

    compile_kwargs = {}
    if config['model_type'] == 'vqc':
        compile_kwargs["curry"] = config["compiler"].get("curry", False)
    compiled_train = ansatz.compile_dataset(df_train, **compile_kwargs)
    compiled_val = ansatz.compile_dataset(df_val, **compile_kwargs)
    print(f" Dataset footprints compiled: Train={len(compiled_train)} | Val={len(compiled_val)}")

    # Model initialisation phase
    log_phase("Initializing Model Parameters")
    if hasattr(text_model, "from_symbols"):
        cols = [col for col in compiled_train.columns if col.endswith('_symbols')]
        txt_stream = []
        for col in cols:
            txt_stream += compiled_train[col].tolist() + compiled_val[col].tolist()
        
        sym_kwargs = {"id_init": True} if config["model_type"] == "vqc" else {}
        text_model.from_symbols(txt_stream, **sym_kwargs)
        print(f" Text Model vocabulary locked: {len(text_model.symbols)} distinct symbols.")

    if hasattr(text_model, "from_plans"):
        cols = [col for col in compiled_train.columns if col.endswith('_einsum')]
        plan_stream = []
        for col in cols:
            plan_stream += compiled_train[col].tolist() + compiled_val[col].tolist()
        text_model.from_plans(list(plan_stream))
        print(f" Text Model Parameters mapped: {len(text_model.leaves)} Leaves | {len(text_model.mlps)} MLPs.")

    if hasattr(image_model, "fit_image_pca") and config['vision']['neural'] == False:
        train_embeddings = torch.load(config['splits']['train']['img_path'])
        image_model.fit_image_pca(torch.stack(list(train_embeddings.values())).to(DEV))
        print(" Visual projection layers calibrated via target PCA.")
    print(f" Model Parameter Counts: Image={sum(p.numel() for p in image_model.parameters())} | Text={sum(p.numel() for p in text_model.parameters())}")

    # Prepare data loaders and optimisers
    log_phase("Preparing Pipeline Execution")
    if config['vision']['use_clip']:
        img_transform = None
    else:
        img_transform = v2.Compose([v2.ToImage(),               
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Resize((64, 64))])

    train_dataset = DatasetClass(compiled_train, config['splits']['train']['img_path'], image_transform=img_transform, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = DatasetClass(compiled_val, config['splits']['val']['img_path'], image_transform=img_transform, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    print(f" Batched steps mapped: Train={len(train_loader)} steps | Val={len(val_loader)} steps")

    optimizer = torch.optim.Adam(list(text_model.parameters()) + list(image_model.parameters()), lr=config['learning_rate'])
    trainer = ContrastiveTrainer(image_model, text_model, optimizer, loss_fn, DEV)
    evaluator = MMEvaluator(image_model, text_model, DEV)
    print(" Gradient step managers and performance metrics trackers bound.")

    # Serialisation and logging setup
    txt_tower = type(text_model).__name__
    img_tower = type(image_model).__name__
    run_name = gen_id(config)

    checkpoint_dir = Path(f"./checkpoints/{DATASET}/{txt_tower}_{img_tower}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    with open(checkpoint_dir / f"{run_name}_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    mlf_db_path = ROOT_PATH / f"mlf_dbs/{DATASET}.db"
    mlf_db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.pytorch.autolog(log_models=False)
    mlflow.set_tracking_uri(f"sqlite:///{mlf_db_path}")
    mlflow.set_experiment(DATASET)

    hostname = socket.gethostname()
    system_node = platform.node()

    # Training and evaluation loop
    log_phase(f"Model #{run_name}: optimisation and evaluation started on \"{hostname}\"...")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": config['epochs'],
            "batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "temperature_parameter": loss_fn.temperature,
            "device_target": str(DEV),
            "seed": SEED,
            "text_tower": txt_tower,
            "image_tower": img_tower,
            "execution_host": hostname,
            "platform_node": system_node,
            "model_path": str(checkpoint_path),
            })
        
        epoch_pbar = tqdm(range(config['epochs']), desc="Training Pipeline", unit="epoch")

        for epoch in epoch_pbar:
            start_time = time.time()
            loss = trainer.train_epoch(train_loader, eval_mapper)
            elapsed_time = time.time() - start_time

            metrics = {}
            tasks = config.get('evaluation_tasks', ['global_retrieval'])
            for task_name in tasks:
                eval_fn = getattr(evaluator, task_name, None)
                if eval_fn is None:
                    print(f"Warning: Evaluation method '{task_name}' not found on MMEvaluator. Skipping.")
                    continue

                if task_name == "global_retrieval":
                    task_metrics = eval_fn(val_loader, eval_mapper)
                elif task_name == "compositional_diagnostic":
                    task_metrics = eval_fn(val_loader, labels=config['dataset']['labels'])
                else:
                    task_metrics = eval_fn(val_loader)
                metrics.update(task_metrics)
            
            mlflow.log_metrics(metrics, step=epoch)
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            tqdm.write(f" Epoch {epoch:02d} | Loss: {loss:.4f} | {metrics_str} | Time: {elapsed_time:.1f}s")

            torch.save({
                "image": image_model.state_dict(),
                "text": text_model.state_dict(),
                "epoch": epoch,
                "train_loss": loss,
                "val_metrics": metrics
                }, checkpoint_path)
            
    log_phase("Experiment Run Concluded")