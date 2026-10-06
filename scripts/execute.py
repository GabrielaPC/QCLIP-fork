import argparse
from pathlib import Path
import pandas as pd

from modules.utils.factory import build_experiment
from modules.utils.general import log_phase, setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
# from modules.compilation.quantum.emul import BackendManager, Emulator
from modules.compilation.quantum.hard_run import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('-cp',"--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt file)")
    parser.add_argument('-l', "--limit", type=int, default=None, help="Debug Flag: Truncate execution to top N samples")
    args = parser.parse_args()

    config, DEV, _ = setup_exp(args.config)

    log_phase("Evaluation Environment")
    print(f" Target Device   : {DEV} ")

    log_phase("Instantiating Architecture")
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)

    data_engine = DataEngine(config, ansatz, DEV)
    compiled_train = data_engine.compile_text('train')
    compiled_val = data_engine.compile_text('val')
    emulation_set = config['emulate']['set']
    compiled_eval = data_engine.compile_text(emulation_set)
    einsum_data = data_engine.describe_einsum(text_model, compiled_eval, return_metrics=True)
    if config['emulate']['measurement_method'] == 'compute_uncompute':
        qlimit = einsum_data['max'][0] # (einsum_data['max'][0] + einsum_data['avg'][0]) // 2
    else:
        qlimit = einsum_data['max'][0] + config['embedding_qubits']

    log_phase("Restoring Dynamic Parameter Spaces")
    data_engine.text_init(text_model, pd.concat([compiled_train, compiled_val], ignore_index=True))
    data_engine.image_init(image_model)

    log_phase("Loading Model Checkpoint Weights")
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        checkpoint = CheckpointManager.load_model(checkpoint_path, image_model, text_model, DEV)
        print(f" Recovered from ({checkpoint_path.name}) Epoch: {checkpoint.get('epoch', 'N/A')} | Historical Loss: {checkpoint.get('train_loss', 'N/A')}")

    image_model.eval()
    text_model.eval()

    provider_name = config.get("backend")

    if provider_name == "IonQ":
        backend_manager = IonQBackendManager(config, qlimit)
    else:
        backend_manager = ProviderBackendManager(config, qlimit)

    log_phase("Extracting Parameter Angle Maps")
    eval_dataset = data_engine.get_dataset(compiled_eval, emulation_set)
    pos_circs, neg_circs = backend_manager.compile_circuits(eval_dataset, text_model, image_model, args.limit, opt_lvl=config['emulate']['optimization_level'])

    # 8. Execution and metric compiling
    log_phase("Executing Emulator Engine Pipeline")
    if provider_name == "IonQ":
        provider = IonQProviderManager(config, backend_manager.backend)
    else:
        provider = ProviderManager(config, backend_manager.backend)

    max_nq = max(qc.num_qubits for qc in pos_circs + neg_circs)
    nq_out = config['embedding_qubits']
    provider.shot_estimation(nq_out, max_nq - nq_out, resolution=config['emulate']['resolution'])
    print(f" Simulating quantum states across {len(pos_circs)} pairs with {provider.shots} shot resolution")

    circs = []

    for i in range(len(pos_circs)):
        circs.append(pos_circs[i])
        circs.append(neg_circs[i])

    data = provider.submit_job(circs)


    # log_phase("Benchmark Emulation Results")
    # print(f" {'Metric Key':<35} | {'Value / Score':<25}")
    # print(" " + "—" * 63)
    # print(f"  {'Evaluated Dataset Pairs':<34} | {len(pos_circs):.6f}")
    # print(f"  {'Evaluation Runs':<34} | {num_runs}")
    # print(f"  {'Pipeline Cumulative Accuracy':<34} | \033[92m{data['accuracy']:.6f}\033[0m")
    
    # if num_runs > 1:
    #     print(f"  {'Accuracy Std Deviation':<34} | ±{data['accuracy_std']:.6f}")
    #     print(f"  {'Accuracy Sample Variance':<34} | {data['accuracy_var']:.6f}")
    #     print(f"  {'Accuracy 95% Conf. Interval':<34} | ±{data['accuracy_ci95']:.6f}")
    #     print(f"  {'Pos Pair Fidelity (Mean ± Std)':<34} | {data['pos_fidelity_mean']:.4f} ± {data['pos_fidelity_std']:.4f}")
    #     print(f"  {'Neg Pair Fidelity (Mean ± Std)':<34} | {data['neg_fidelity_mean']:.4f} ± {data['neg_fidelity_std']:.4f}")
    # print(" " + "—" * 63 + "\n")

if __name__ == "__main__":
    main()