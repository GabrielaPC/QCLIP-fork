import torch, argparse
import pandas as pd

from modules.utils.factory import build_experiment
from modules.utils.general import log_phase, setup_exp
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import ContrastiveTrainer, MMEvaluator, RunManager

# uv run python train.py --config configs/tensor_network.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', "--config", type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    config, DEV, SEED = setup_exp(args.config)
    DATASET = config['dataset']['name']
    BATCH_SIZE = config['batch_size']

    log_phase("Environment Initialized")
    print(f" Target Device : {DEV} | Random Seed : {SEED} | Run ID : {DATASET}")

    log_phase("Setting Up Experiment Components")
    ansatz, image_model, text_model, loss_fn = build_experiment(config, DEV)

    log_phase("Compiling Symbolic Datasets")
    data_engine = DataEngine(config, ansatz, DEV)
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')
    print(f" Datasets compiled: Train={len(compiled_train)} | Val={len(compiled_val)}")
    data_engine.describe_einsum(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.describe_einsum(image_model)

    log_phase("Initializing Model Parameters")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)
    print(f" Model Parameter Counts: Image={sum(p.numel() for p in image_model.parameters()):,} | Text={sum(p.numel() for p in text_model.parameters()):,}")

    log_phase("Preparing Pipeline Execution")
    train_loader = data_engine.get_loader(compiled_train, split='train')
    val_loader = data_engine.get_loader(compiled_val, split='val')

    quantum_params = list(text_model.parameters()) + list(image_model.params)
    classical_params = list(image_model.projector.parameters()) if config['vision']['method'] == 'mlp' else []
    optimizer = torch.optim.Adam(
        [{'params': quantum_params, 'lr': config['qlr']},
         {'params': classical_params, 'lr': config['clr']}], 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    trainer = ContrastiveTrainer(image_model, text_model, optimizer, loss_fn, DEV)
    evaluator = MMEvaluator(image_model, text_model, DEV)

    manager = RunManager(config, trainer, evaluator, DEV, SEED)
    manager.fit(train_loader, val_loader, data_engine.eval_mapper)