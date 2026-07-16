import argparse, sys, torch
from pathlib import Path
import pandas as pd

from modules.utils.factory import  build_experiment

from modules.utils.factory import build_experiment
from modules.utils.general import log_phase, setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager, Emulator


# Framework path setup
ROOT_PATH = Path.cwd()
sys.path.insert(0, str(ROOT_PATH))

def circ_acc(pos_f, neg_f):
    return (torch.sum((pos_f > neg_f)) / len(pos_f)).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('-cp',"--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument('-l', "--limit", type=int, default=None, help="Debug Flag: Truncate execution to top N samples")
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
    emulation_set = config['emulation']['set']
    compiled_eval = data_engine.compile_text(emulation_set)
    einsum_data = data_engine.describe_einsum(text_model, compiled_eval, return_metrics=True)
    qlimit = einsum_data['avg'][0] if config['dataset']['name'] == 'aro' else einsum_data['max'][0]

    log_phase("Restoring Dynamic Parameter Spaces")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)

    log_phase("Loading Model Checkpoint Weights")
    checkpoint = CheckpointManager.load_model_weights(checkpoint_path, image_model, text_model, DEV)
    print(f" Recovered from Epoch: {checkpoint.get('epoch', 'N/A')} | Historical Loss: {checkpoint.get('train_loss', 'N/A')}")

    image_model.eval()
    text_model.eval()
    backend_manager = BackendManager(config, DEV, qlimit)

    log_phase("Extracting Parameter Angle Maps")
    txt_params_dict = text_model._get_params()
    img_params_dict = image_model._get_params() 
    pos_circs, neg_circs = backend_manager.compile_circuits(compiled_val, txt_params_dict, img_params_dict)

    # 8. Execution and metric compiling
    log_phase("Executing Emulator Engine Pipeline")
    emulator = Emulator(config, DEV, qlimit)
    max_nq = max(qc.num_qubits for qc in pos_circs + neg_circs)
    nq_out = config['embedding_qubits']
    emulator.shot_estimation(nq_out, max_nq - nq_out, epsilon=config['eps'])
    print(f" Simulating quantum states across {len(pos_circs)} pairs with {emulator.shots} shot resolution")
    data = emulator.run_experiment(pos_circs, neg_circs, batch_size=64)
    acc = sum(data['correct']) / len(data['correct'])

    log_phase("Benchmark Emulation Results")
    print(f" {'Metric Key':<35} | {'Value / Score':<15}")
    print(" " + "—" * 53)
    print(f"  {'Evaluated Dataset Pairs':<34} | {len(pos_circs):.6f}")
    print(f"  {'Pipeline Cumulative Accuracy':<34} | \033[92m{acc:.6f}\033[0m")
    print(" " + "—" * 53 + "\n")

if __name__ == "__main__":
    main()