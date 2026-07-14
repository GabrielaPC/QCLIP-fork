from concurrent.futures import ThreadPoolExecutor
import argparse, logging, math, os, sys, time, yaml, torch, traceback
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2

# Qiskit suite imports
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeMiami # FakeMiamiV2
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnrollCustomDefinitions, BasisTranslator
from qiskit.transpiler import CouplingMap
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
#from qiskit.transpiler.passes import RemoveIdleWires

from modules.data_pipeline.datasets import SwapDataset, SVODataset, ARODataset
from modules.utils.quantum_ops import tn2qiskit, amplitude_encoding
from modules.utils.tensor_ops import einsum2interleaved
from modules.utils.general import load_pkl, get_device, store_pkl
from modules.utils.analysis import tn_metadata, analyse_einsum
from factory import build_dataset, build_experiment
from modules.utils.general import gen_id


# Framework path setup
ROOT_PATH = Path.cwd()
sys.path.insert(0, str(ROOT_PATH))

def _run_circuits(qc_array, backend, batch_size=32):
    result_array = []
    indices = range(len(qc_array))
    chunks = [indices[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]
    
    submitted_jobs = []
    for idx_list in chunks:
        chunk_circs = [qc_array[i] for i in idx_list]
        job = backend.run(chunk_circs, shots=1) 
        submitted_jobs.append((idx_list, job))
        
    for idx_list, job in tqdm(submitted_jobs, desc="Analysing Density Matrices"):
        result = job.result()
        for i, idx in enumerate(idx_list):
            rho = result.data(i)["rho_out"]
            matrix = rho.data if hasattr(rho, "data") else rho
            fidelity = float(np.real(matrix[0, 0]))
            fidelity = np.clip(fidelity, 0.0, 1.0)
            result_array.append(0.5 + (0.5 * fidelity))
            
    result_tensor = torch.tensor(result_array)
    return torch.full(result_tensor.size(), torch.pi / 2) - torch.acos(result_tensor.abs().clamp(0, 1))

def run_circuits(qc_array, backend, shots, batch_size=24):
    result_array = []
    indices = range(len(qc_array))
    chunks = [indices[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]

    for idx_list in tqdm(chunks, desc="Running Circuits"):
        chunk_circs = [qc_array[i] for i in idx_list]
        job = backend.run(chunk_circs, shots=shots)
        result = job.result()
        for i, idx in enumerate(idx_list):
            qc = qc_array[idx]
            counts = result.get_counts(i)

            output_qubits = qc.metadata.get("output_qubits", [])
            num_clbits = qc.num_clbits
            zero_str = "0" * num_clbits

            check_indices = [num_clbits - 1 - b for b in range(num_clbits) if b not in output_qubits]
            accepted_shots = 0
            all_zero_shots = 0

            for bitstring, count in counts.items():
                if all(bitstring[idx] == '0' for idx in check_indices):
                    accepted_shots += count
                    if bitstring == zero_str:
                        all_zero_shots += count

            fidelity = all_zero_shots / accepted_shots if accepted_shots > 0 else 0.0
            result_array.append(0.5 + (0.5 * fidelity))
            
    result_tensor = torch.tensor(result_array)
    return torch.full(torch.asin(result_tensor.abs().clamp(0,1)))
    #return torch.full(result_tensor.size(), torch.pi / 2) - torch.acos(result_tensor.abs().clamp(0, 1))


def batched_transpile(circuits, coupling_map, basis_gates, batch_size=128, optimization_level=1):
    transpiled_circs = []
    chunks = [circuits[i:i + batch_size] for i in range(0, len(circuits), batch_size)]
    # remove_idle_pm = PassManager([RemoveIdleWires()])
    for chunk in tqdm(chunks, desc="Transpiling", unit="batch"):
        transpiled_chunk = transpile(
            chunk, 
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            num_processes=None 
        )
        # transpiled_chunk = [remove_idle_pm.run(qc) for qc in transpiled_chunk]
        transpiled_circs.extend(transpiled_chunk)
    return transpiled_circs

def shot_estimation(nq_out, nq_ps, epsilon=0.01):    
    req_shots = 0.25 / (epsilon ** 2)
    raw_shots = req_shots * (2 ** nq_ps)
    min_shots = (2 ** nq_out) * 10
    final_shots = max(raw_shots, min_shots)
    final_shots = int(math.ceil(final_shots))
    return max(4096, min(final_shots, 1_000_000))

def circ_acc(pos_f, neg_f):
    return (torch.sum((pos_f > neg_f)) / len(pos_f)).item()

def log_phase(name: str):
    print(f"\n [{name.upper()}] " + "—" * (60 - len(name)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument('-cp',"--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument('-l', "--limit", type=int, default=None, help="Debug Flag: Truncate execution to top N samples")
    args = parser.parse_args()

    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    
    with open(args.config, "r") as file:
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
    emulation_set = config['emulation_set']    
    df_eval = load_pkl(config['splits']['test'][emulation_set]["text_path"])

    compile_kwargs = {}
    if config["model_type"] == "vqc":
        compile_kwargs["curry"] = config["text"].get("curry", False)
    compiled_eval = ansatz.compile_dataset(df_eval, **compile_kwargs)

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
        text_model.from_symbols(symbol_arr)

        einsum_cols = [c for c in compiled_train.columns if c.endswith("_einsum")]
        einsum_arr = []
        for col in einsum_cols:
            einsum_arr += compiled_train[col].tolist() + compiled_val[col].tolist() 


        tn_arr = list(zip(einsum_arr, symbol_arr))
        metrics = tn_metadata(tn_arr)
        print(f"Circuit Metrics for Text Model:")
        print(f"(Max) Qubits: {metrics['max'][0]:.4f} | Gates: {metrics['max'][1]:.4f} | Depth: {metrics['max'][2]:.4f} | Rank: {metrics['max'][3]:.4f}")
        print(f"(Avg) Qubits: {metrics['avg'][0]:.4f} | Gates: {metrics['avg'][1]:.4f} | Depth: {metrics['avg'][2]:.4f} | Rank: {metrics['avg'][3]:.4f}")
        symbol_arr, einsum_arr = [], []
        for scol, ecol in zip(sym_cols, einsum_cols):
            symbol_arr += compiled_eval[scol].tolist()
            einsum_arr += compiled_eval[ecol].tolist()
        eval_tn_arr = list(zip(einsum_arr, symbol_arr))
        metrics = tn_metadata(eval_tn_arr)
        print(f"Circuit Metrics for Evaluation Set:")
        print(f"(Max) Qubits: {metrics['max'][0]:.4f} | Gates: {metrics['max'][1]:.4f} | Depth: {metrics['max'][2]:.4f} | Rank: {metrics['max'][3]:.4f}")
        print(f"(Avg) Qubits: {metrics['avg'][0]:.4f} | Gates: {metrics['avg'][1]:.4f} | Depth: {metrics['avg'][2]:.4f} | Rank: {metrics['avg'][3]:.4f}")
        qlimit = metrics['avg'][0] if config['dataset']['name'] == 'aro' else metrics['max'][0]

    if config['vision']['method'] == 'pca':
        train_embeddings = torch.load(config['splits']['train']['img_path'])
        image_model.fit_image_pca(torch.stack(list(train_embeddings.values())).to(DEV))
    if config['vision']['method'] in ['mlp', 'pca']:
        metrics = analyse_einsum(image_model.einsum_expr.replace('b', ''), image_model.gate_arr)
        print(f"Circuit Metrics for Image Model:")
        print(f"Qubits: {metrics[0]:.4f} | Gates: {metrics[1]:.4f} | Depth: {metrics[2]:.4f} | Rank: {metrics[3]:.4f}")

    # 2. Extract and Parse Trained Classical-Quantum Weights
    log_phase("Loading Model Checkpoint Weights")
    checkpoint = torch.load(args.checkpoint, map_location=DEV)
    img_params = sum(p.numel() for p in checkpoint["image"].values() if hasattr(p, "numel"))
    txt_params = sum(p.numel() for p in checkpoint["text"].values() if hasattr(p, "numel"))
    print(f" Restored State Dict -> Image Params: {img_params:,} | Text Params: {txt_params:,}")

    image_model.load_state_dict(checkpoint["image"])
    text_model.load_state_dict(checkpoint["text"])

    saved_epoch = checkpoint.get("epoch", "N/A")
    saved_loss = checkpoint.get("train_loss", "N/A")
    print(f"    [Checkpoint Info] -> Recovered from Epoch: {saved_epoch} | Historical Loss: {saved_loss}")

    image_model.eval()
    text_model.eval()

    log_phase("Quantum Emulator Backend Setup")
    devices = AerSimulator().available_devices()
    print(f" Available AerSimulator Devices: {', '.join(devices)}")
    qdev = 'GPU' if 'GPU' in devices else 'CPU'
    method = 'statevector' # 'statevector', 'tensor_network', 'matrix_product_state', 'density_matrix'
    backend_options = {
        "device": qdev,
        "method": method,
        "precision": "single",

        "max_parallel_experiments": 0,
        "max_parallel_threads": 0,
        "max_parallel_shots": 0,

        "fusion_enable": True,
        "fusion_threshold": 2,
        "fusion_max_qubit": 6,
    }

    hardware_profile = FakeMiami()
    miami_edges = hardware_profile.coupling_map.get_edges()
    filtered_edges = [edge for edge in miami_edges if edge[0] < qlimit and edge[1] < qlimit]
    coupling_map = CouplingMap(filtered_edges)
    basis_gates = hardware_profile.basis_gates

    if config['noise']:
        noise_model = NoiseModel.from_backend(hardware_profile)
        backend_options["noise_model"] = noise_model 

        backend_options['shot_branching_enable'] = True
        backend_options["seed_simulator"] = 42
        backend_options["statevector_parallel_threshold"] = 0

        if method == 'statevector':
            backend_options["batched_shots_gpu"] = True
            backend_options["max_parallel_shots"] = 256
    else:
        if method == 'statevector':
            backend_options["statevector_sample_measure_opt"] = 10
            backend_options["batched_shots_gpu"] = True
            backend_options["max_parallel_shots"] = 256
    print(f" Simulation: Profile {hardware_profile.name} ({qlimit} qubits), Device {qdev}, Method {method}, Noise {'Enabled' if config['noise'] else 'Disabled'}")

    backend = AerSimulator(method=method, device=qdev, cuStateVec_enable=True) 
    backend.set_options(**backend_options)

    # basis_gates = backend.operation_names
    # pm = PassManager([
    #     UnrollCustomDefinitions(sel, basis_gates=basis_gates),
    #     BasisTranslator(sel, target_basis=basis_gates)
    # ])

    # 3. Dynamic Symbol Map Generation for Qiskit Translation
    log_phase("Extracting Parameter Angle Maps")
    txt_params_dict = {}
    if hasattr(text_model, "sym2param") and hasattr(text_model, "params"):
        for symbol, idx in text_model.sym2param.items():
            txt_params_dict[symbol] = float(text_model.params[idx].detach().cpu().item())
            
    img_params_dict = {}
    use_ansatz = config['vision']['method'] != 'amp'
    if use_ansatz and hasattr(image_model, "sym2param") and hasattr(image_model, "params"):
        for symbol, idx in image_model.sym2param.items():
            img_params_dict[symbol] = float(image_model.params[idx].detach().cpu().item())
        print("Using ansatz for image encoding.")
    else:
        print("Using CLIP for image encoding with amplitude encoding.")

    print(f" Mapped parameters directly from live state: {len(txt_params_dict)} Text | {len(img_params_dict)} Vision.")

    log_phase("Constructing Data Engine Context")
    if config['vision']['method'] == 'amp':
        img_transform = None
    else:
        img_transform = v2.Compose([v2.ToImage(),               
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Resize((64, 64))])
    
    if config['dataset']['name'] == 'svo-probes':
        if emulation_set == 'swap':
            DatasetClass = SwapDataset
        # else:
        #     DatasetClass = SVOProbesDataset
    elif config['dataset']['name'] == 'aro':
        DatasetClass = ARODataset

    dataset = DatasetClass(compiled_eval, config["splits"]['test'][emulation_set]["img_path"], image_transform=img_transform)
    
    # 4. Assemble Circuit Suite using Evaluation Dataframe
    log_phase("Compiling Swap-Test Circuits")
    raw_pos_circs = []
    raw_neg_circs = []
    pos_metadata = []
    neg_metadata = []
    pos_param_maps = []
    neg_param_maps = []

    data_size = min(args.limit, len(dataset)) if args.limit else len(dataset)
    failed_circuits = 0
    failed_samples = []
    for idx in tqdm(range(data_size), desc="Compiling Qiskit DAGs", unit="pair"):
        try:
            sample = dataset[idx]
            image = sample["image"]
            pos_einsum, pos_wires = sample["pos_caption"][0], sample["pos_caption"][1]
            neg_einsum, neg_wires = sample["neg_caption"][0], sample["neg_caption"][1]

            # 1. Generate Text Subcircuit Layout
            qc_pos_txt, pos_output_qubits, pos_txt_params = tn2qiskit(einsum2interleaved(pos_einsum), pos_wires, txt_params_dict, False)
            qc_neg_txt, neg_output_qubits, neg_txt_params = tn2qiskit(einsum2interleaved(neg_einsum), neg_wires, txt_params_dict, False)

            if qc_pos_txt.num_qubits > qlimit or qc_neg_txt.num_qubits > qlimit:
                failed_circuits += 1
                failed_samples.append(idx)
                #print(f" Circuit compilation dropped at sample index {idx}: Exceeds max qubit limit ({max_nq})")
                continue

            if use_ansatz:
                img_vars = img_params_dict | image_model.encode_features(image)
                in_idx, out_idx = einsum2interleaved(image_model.einsum_expr.replace('b', ''))
                qc_img, _, img_params = tn2qiskit([in_idx, out_idx], image_model.gate_arr, img_vars, False)
                pos_params = pos_txt_params | img_params
                neg_params = neg_txt_params | img_params
            else:
                img_vec = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
                normed_img_vec = amplitude_encoding(img_vec)
                qc_img = QuantumCircuit(int(math.ceil(math.log2(len(normed_img_vec)))), 0)
                qc_img.initialize(normed_img_vec)
                pos_params, neg_params = pos_txt_params, neg_txt_params

            pos_params = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in pos_params.items()}
            neg_params = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in neg_params.items()}
            qc_img_inv = qc_img.inverse()

            # 3. Assemble frames
            qc_pos = QuantumCircuit(qc_pos_txt.num_qubits, qc_pos_txt.num_clbits)
            qc_pos.compose(qc_pos_txt, inplace=True)
            qc_pos.compose(qc_img_inv, qubits=pos_output_qubits, inplace=True)
            qc_pos.measure(pos_output_qubits, pos_output_qubits)
            # qc_pos.save_density_matrix(qubits=pos_output_qubits, label="rho_out")

            qc_neg = QuantumCircuit(qc_neg_txt.num_qubits, qc_neg_txt.num_clbits)
            qc_neg.compose(qc_neg_txt, inplace=True)
            qc_neg.compose(qc_img_inv, qubits=neg_output_qubits, inplace=True)
            qc_neg.measure(neg_output_qubits, neg_output_qubits)
            # qc_neg.save_density_matrix(qubits=neg_output_qubits, label="rho_out")

            raw_pos_circs.append(qc_pos)
            raw_neg_circs.append(qc_neg)
            pos_metadata.append({"output_qubits": pos_output_qubits})
            neg_metadata.append({"output_qubits": neg_output_qubits})
            pos_param_maps.append(pos_params)
            neg_param_maps.append(neg_params)
            
        except Exception as e:
            failed_samples.append(idx)
            failed_circuits += 1
            #tqdm.write(f" Circuit compilation dropped at sample index {idx}: {e}")
    print(f" Compiled {len(raw_pos_circs)} circuit pairs | Dropped {failed_circuits} samples due to errors or qubit limits")
    print(failed_samples)
    print(f"\n[CPU] Transpiling {len(raw_pos_circs) * 2} circuits in parallel across all cores...")
    # pos_circs = transpile(raw_pos_circs, backend=backend, optimization_level=2, num_processes=0)
    # neg_circs = transpile(raw_neg_circs, backend=backend, optimization_level=2, num_processes=0)
    pos_circs = batched_transpile(raw_pos_circs, coupling_map, basis_gates, batch_size=256, optimization_level=2)
    neg_circs = batched_transpile(raw_neg_circs, coupling_map, basis_gates, batch_size=256, optimization_level=2)

    for i in range(len(pos_circs)):
        pos_circs[i].metadata = pos_metadata[i]
        pos_circs[i].assign_parameters(pos_param_maps[i], inplace=True)
        
        neg_circs[i].metadata = neg_metadata[i]
        neg_circs[i].assign_parameters(neg_param_maps[i], inplace=True)

    log_phase("Executing Emulator Engine Pipeline")
    max_nq = max(qc.num_qubits for qc in pos_circs + neg_circs)
    nq_out = config['embedding_qubits']
    shots = shot_estimation(nq_out, max_nq - nq_out, epsilon=config['eps'])
    print(f" Simulating quantum states across {len(pos_circs)} pairs with {shots} shot resolution")

    start_eval_time = time.time()
    with torch.no_grad():
        pos_f = run_circuits(pos_circs, backend, shots, 64) 
        neg_f = run_circuits(neg_circs, backend, shots, 64) 
        cum_acc = circ_acc(pos_f, neg_f)

        run_type = "noisy" if config.get("noise") else "noiseless"
        run_name = gen_id(config)
        if run_type == "noisy":
            save_filename = f"results/{config['dataset']['name']}/{run_name}_{run_type}{config['eps']}_margins.pkl"
        else:
            save_filename = f"results/{config['dataset']['name']}/{run_name}_{run_type}_margins.pkl"

        def to_ndarray(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().flatten()
            return np.asarray(x).flatten()

        pos_arr = to_ndarray(pos_f)
        neg_arr = to_ndarray(neg_f)
        margins = pos_arr - neg_arr
        correct = margins > 0

        data = {
            "pos_score": pos_arr,
            "neg_score": neg_arr,
            "margin": margins,
            "correct": correct
        }
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        store_pkl(data, save_filename)
        print(f" Saved emulation results to {save_filename}")
    elapsed = time.time() - start_eval_time
    print(f"    Completed in {elapsed:.2f}s")

    log_phase("Benchmark Emulation Results")
    print(f" {'Metric Key':<35} | {'Value / Score':<15}")
    print(" " + "—" * 53)
    print(f"  {'Evaluated Dataset Pairs':<34} | {len(pos_circs):.6f}")
    print(f"  {'Target Resolution (Shots)':<34} | {shots:.6f}")
    print(f"  {'Pipeline Cumulative Accuracy':<34} | \033[92m{cum_acc:.6f}\033[0m")
    print(" " + "—" * 53 + "\n")

if __name__ == "__main__":
    main()