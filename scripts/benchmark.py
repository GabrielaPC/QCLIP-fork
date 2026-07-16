import argparse, time, torch
from pathlib import Path
import pandas as pd

# Local framework imports
from modules.utils.factory import build_experiment
from modules.models.fusion.engine import MMEvaluator
from modules.utils.general import log_phase, setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument("-cp", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    args = parser.parse_args()

    config, DEV, _ = setup_exp(args.config)
    checkpoint_path = Path(args.checkpoint)

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV} | Weights : {checkpoint_path.name}")

    log_phase("Instantiating Architecture")
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)

    data_engine = DataEngine(config, ansatz, DEV)
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')

    log_phase("Restoring Dynamic Parameter Spaces")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)

    log_phase("Loading Model Checkpoint Weights")
    checkpoint = CheckpointManager.load_model_weights(checkpoint_path, image_model, text_model, DEV)
    print(f" Recovered from Epoch: {checkpoint.get('epoch', 'N/A')} | Historical Loss: {checkpoint.get('train_loss', 'N/A')}")

    image_model.eval()
    text_model.eval()
    evaluator = MMEvaluator(image_model, text_model, DEV)

    log_phase("Compiling Evaluation Graph Structures")
    test_sets = config["splits"]["test"]
    results = {}
    for split_name, split_info in test_sets.items():
        compiled_eval = data_engine.compile_text(split_name)
        data_engine.describe_einsum(text_model, compiled_eval)
        eval_loader = data_engine.get_loader(compiled_eval, split=split_name)
        eval_fn = getattr(evaluator, split_info['task'], None)

        with torch.no_grad():
            start_task = time.time()
            task_metrics = eval_fn(eval_loader)
            elapsed = time.time() - start_task
            print(f" -> Evaluated '{split_name}' task in {elapsed:.2f}s")
            for key, val in task_metrics.items(): results[f"{split_name}_{key}"] = val

    log_phase("Final Benchmark Scoreboard")
    print(f"{'Split & Metric Key':<35} | {'Value / Score':<15}")
    print("—" * 53)
    for key, val in results.items():
        print(f"{key:<35} | {val:.6f}")
    print("—" * 53 + "\n")