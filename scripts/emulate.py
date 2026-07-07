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
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

from modules.data_pipeline.datasets import SwapDataset
from modules.utils.quantum_ops import tn2qiskit, amplitude_encoding
from modules.utils.tensor_ops import einsum2interleaved
from modules.utils.general import load_pkl, get_device
from modules.utils.analysis import tn_metadata, analyse_einsum
from factory import build_dataset, build_experiment
import concurrent

# Framework path setup
ROOT_PATH = Path.cwd()
sys.path.insert(0, str(ROOT_PATH))

def run_circuits(qc_array, backend, shots, batch_size=1, tout=30):
    result_array = []
    indices = range(len(qc_array))
    chunks = [indices[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for idx_list in tqdm(chunks, desc="Simulating Chunks"):
            chunk_circs = [qc_array[i] for i in idx_list]

            def execute_chunk():
                job = backend.run(chunk_circs, shots=shots)
                return job.result()
            future = executor.submit(execute_chunk)

            try:
                result = future.result(timeout=tout)
                for i, idx in enumerate(idx_list):
                    counts = result.get_counts(i)
                    total_shots = sum(counts.values())
                    meas_qubits = len(next(iter(counts.keys())))
                    all_zero_shots = counts.get("0" * meas_qubits, 0)
                    fidelity = all_zero_shots / total_shots
                    result_array.append(0.5 + (0.5 * fidelity))
            except concurrent.futures.TimeoutError:
                print(f"\n[Warning] Chunk with indices {idx_list} timed out after {tout}s. Skipping...")
                continue
            except Exception as e:
                print(f"\n[Error] Chunk with indices {idx_list} failed due to: {e}. Skipping...")
                continue
            
    result_tensor = torch.tensor(result_array)
    return torch.full(result_tensor.size(), torch.pi / 2) - torch.acos(result_tensor.abs().clamp(0, 1))


def safe_batch_size(circuits, vram_gb=24, safety_factor=0.2):
    if not circuits:
        return 1
    max_qubits = max(c.num_qubits for c in circuits)
    statevector_bytes = (2 ** max_qubits) * 16
    copies_per_circuit = 2
    mem_per_circuit = statevector_bytes * copies_per_circuit
    usable_memory = vram_gb * 1e9 * safety_factor
    batch = max(1, int(usable_memory / mem_per_circuit))
    return min(batch, 64)

def _run_circuits(qc_array, sampler, shots):
    result_array = []

    print(f"Starting serial execution of {len(qc_array)} circuits on GPU...")
    
    # Process each quantum circuit individually to prevent GPU VRAM saturation
    for qc in tqdm(qc_array, desc="Simulating Circuits"):
        # Submit a single circuit as a list containing one element
        job = sampler.run([qc], shots=shots)
        primitive_result = job.result()
        
        # Extract the result for this single execution block
        pub_result = primitive_result[0]
        
        reg_name = list(pub_result.data.keys())[0]
        counts = getattr(pub_result.data, reg_name).get_counts()

        meas_qubits = len(next(iter(counts.items()))[0])
        post_selected_shots = counts.get("0" * (meas_qubits), 1)
        
        try:
            failed_state_counts = counts.get(f'1{"0" * (meas_qubits - 1)}', 1)
            ratio = post_selected_shots / (post_selected_shots + failed_state_counts)
            result_array.append(ratio)
        except Exception:
            result_array.append(0.0)

    # Convert the collected results back into a PyTorch tensor
    result_array = torch.tensor(result_array)
    
    return torch.full(result_array.size(), torch.pi / 2) - torch.acos(result_array.abs().clamp(0, 1))

def circ_acc(pos_f, neg_f):
    return (torch.sum((pos_f > neg_f)) / len(pos_f)).item()

def log_phase(name: str):
    print(f"\n [{name.upper()}] " + "—" * (60 - len(name)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--limit", type=int, default=None, help="Debug Flag: Truncate execution to top N samples")
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
    test_set = config['emulation_set']    
    df_eval = load_pkl(config['splits'][test_set]["text_path"])

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
    backend_options = {
        "device": "GPU",
        "method": "statevector",
        "precision": "single",
        # "batched_shots_gpu": True,
        "max_parallel_experiments": 1,
        "max_parallel_threads": 1,
        "max_parallel_shots": 1,
        "fusion_enable": True,
        "fusion_threshold": 4,
        "fusion_max_qubit": 5,
    }

    if config['noise']:
        hardware_profile = FakeMiami()
        noise_model = NoiseModel.from_backend(hardware_profile)
        backend_options["noise_model"] = noise_model 
        backend_options['shot_branching_enable'] = True
        backend_options["seed_simulator"] = 42
        backend_options["statevector_parallel_threshold"] = 0
        # backend_options["method"] = "density_matrix"
        coupling_map = hardware_profile.coupling_map
        basis_gates = noise_model.basis_gates

        print(" Target Backend: cuQuantum (NOISY / Density Matrix)")
        print(f" Hardware Profile: FakeMiami ({hardware_profile.num_qubits} qubits mapped)")
    else:
        coupling_map = None
        basis_gates = None
        backend_options["method"] = "statevector"
        backend_options["statevector_sample_measure_opt"] = 10
        print(" Target Backend: cuQuantum (NOISELESS / Statevector Graph)")

    
    backend = AerSimulator() # method="tensor_network", device="GPU"
    backend.set_options(**backend_options)

    # sampler.options.backend_options.update(backend_options)

    shots = config.get("shots", 4096)
    print(f" Evaluation Resolution: {shots} shots per circuit execution context")

    basis_gates = backend.operation_names
    pm = PassManager([
        UnrollCustomDefinitions(sel, basis_gates=basis_gates),
        BasisTranslator(sel, target_basis=basis_gates)
    ])

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
        
    swap_dataset = SwapDataset(compiled_eval, config["splits"]['swap']["img_path"], image_transform=img_transform)
    
    # 4. Assemble Circuit Suite using Evaluation Dataframe
    log_phase("Compiling Swap-Test Circuits")
    pos_circs = []
    neg_circs = []
    failed_circuits = 0

    data_size = len(swap_dataset)
    if args.limit:
        data_size = min(args.limit, data_size)
        print(f" Debug Mode Active: Evaluation data limited to top {data_size} items.")

    for idx in tqdm(range(data_size), desc="Compiling Qiskit DAGs", unit="pair"):
        try:
            sample = swap_dataset[idx]
            image = sample["image"]
            pos_einsum, pos_wires = sample["pos_caption"][0], sample["pos_caption"][1]
            neg_einsum, neg_wires = sample["neg_caption"][0], sample["neg_caption"][1]

            # 1. Generate Text Subcircuit Layout
            qc_pos_txt, pos_output_qubits, pos_txt_params = tn2qiskit(
                einsum2interleaved(pos_einsum), pos_wires, txt_params_dict, False
            )
            # for q_idx in range(qc_pos_txt.num_qubits):
            #     if q_idx not in pos_output_qubits: qc_pos_txt.measure(q_idx, q_idx)

            qc_neg_txt, neg_output_qubits, neg_txt_params = tn2qiskit(
                einsum2interleaved(neg_einsum), neg_wires, txt_params_dict, False
            )
            # for q_idx in range(qc_neg_txt.num_qubits):
            #     if q_idx not in neg_output_qubits: qc_neg_txt.measure(q_idx, q_idx)

            if use_ansatz:
                img_vars = img_params_dict | image_model.encode_features(image)
                in_idx, out_idx = einsum2interleaved(image_model.einsum_expr.replace('b', ''))
                qc_img, _, img_params = tn2qiskit([in_idx, out_idx], image_model.gate_arr, img_vars, False)
                pos_params = pos_txt_params | img_params
                neg_params = neg_txt_params | img_params
            else:
                img_vec = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
                normed_img_vec = amplitude_encoding(img_vec)
                required_qubits = int(math.ceil(math.log2(len(normed_img_vec))))
                qc_img = QuantumCircuit(required_qubits, 0)
                qc_img.initialize(normed_img_vec)

                pos_params = pos_txt_params
                neg_params = neg_txt_params


            qc_img_inv = qc_img.inverse()

            # 3. Assemble Positive Swap Test Frame
            qc_pos = QuantumCircuit(qc_pos_txt.num_qubits, qc_pos_txt.num_clbits)
            qc_pos.compose(qc_pos_txt, inplace=True)
            qc_pos.compose(qc_img_inv, qubits=pos_output_qubits, inplace=True)
            qc_pos.measure(pos_output_qubits, range(len(pos_output_qubits)))

            # q_anc_p = QuantumRegister(1, "anc_pos")
            # q_txt_p = QuantumRegister(qc_pos_txt.num_qubits, "txt_pos")
            # q_img_p = QuantumRegister(qc_img.num_qubits, "img_pos")
            
            # c_total_p = ClassicalRegister(qc_pos_txt.num_clbits + 1, "c_total_pos")
            # qc_pos = QuantumCircuit(q_anc_p, q_txt_p, q_img_p, c_total_p)

            # qc_pos.compose(qc_pos_txt, qubits=q_txt_p, inplace=True)
            # qc_pos.compose(qc_img, qubits=q_img_p, inplace=True)

            # qc_pos.h(q_anc_p)
            # for m_idx in range(len(pos_output_qubits)):
            #     qc_pos.cswap(q_anc_p[0], q_txt_p[pos_output_qubits[m_idx]], q_img_p[m_idx])
            # qc_pos.h(q_anc_p)

            # qc_pos.measure(q_anc_p, c_total_p[-1])
            # for q_idx in range(qc_pos_txt.num_qubits):
            #     if q_idx not in pos_output_qubits:
            #         qc_pos.measure(q_txt_p[q_idx], c_total_p[q_idx])

            # 4. Assemble Negative Swap Test Frame
            qc_neg = QuantumCircuit(qc_neg_txt.num_qubits, qc_neg_txt.num_clbits)
            qc_neg.compose(qc_neg_txt, inplace=True)
            qc_neg.compose(qc_img_inv, qubits=neg_output_qubits, inplace=True)
            qc_neg.measure(neg_output_qubits, range(len(neg_output_qubits)))
            # q_anc_n = QuantumRegister(1, "anc_neg")
            # q_txt_n = QuantumRegister(qc_neg_txt.num_qubits, "txt_neg")
            # q_img_n = QuantumRegister(qc_img.num_qubits, "img_neg")

            # c_total_n = ClassicalRegister(qc_neg_txt.num_clbits + 1, "c_total_neg")
            # qc_neg = QuantumCircuit(q_anc_n, q_txt_n, q_img_n, c_total_n)

            # qc_neg.compose(qc_neg_txt, qubits=q_txt_n, inplace=True)
            # qc_neg.compose(qc_img, qubits=q_img_n, inplace=True)
            
            # qc_neg.h(q_anc_n)
            # for m_idx in range(len(neg_output_qubits)):
            #     qc_neg.cswap(q_anc_n[0], q_txt_n[neg_output_qubits[m_idx]], q_img_n[m_idx])
            # qc_neg.h(q_anc_n)

            # qc_neg.measure(q_anc_n, c_total_n[-1])
            # for q_idx in range(qc_neg_txt.num_qubits):
            #     if q_idx not in neg_output_qubits:
            #         qc_neg.measure(q_txt_n[q_idx], c_total_n[q_idx])

            # Transpile directly for backend specifications
            transpiled_pos_qc = transpile(qc_pos, backend=backend, optimization_level=2) # pm.run(qc_pos)
            transpiled_neg_qc = transpile(qc_neg, backend=backend, optimization_level=2) # pm.run(qc_neg)
            transpiled_pos_qc.assign_parameters(pos_params, inplace=True)
            transpiled_neg_qc.assign_parameters(neg_params, inplace=True)
            pos_circs.append(transpiled_pos_qc)
            neg_circs.append(transpiled_neg_qc)
            
        except Exception as e:
            failed_circuits += 1
            tqdm.write(f" Circuit compilation dropped at sample index {idx}: {e}")
            tqdm.write(traceback.format_exc())
            break
    
    # pos_circs = transpile(pos_raw_circs, backend, optimization_level=1)
    # neg_circs = transpile(neg_raw_circs, backend, optimization_level=1)
    log_phase("Executing Emulator Engine Pipeline")
    print(f" Simulating quantum states across {len(pos_circs)} pairs...")

    start_eval_time = time.time()
    with torch.no_grad():
        pos_f = run_circuits(pos_circs, backend, shots) #, safe_batch_size(pos_circs, vram_gb=24))
        neg_f = run_circuits(neg_circs, backend, shots) #, safe_batch_size(neg_circs, vram_gb=24))
        cum_acc = circ_acc(pos_f, neg_f)
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