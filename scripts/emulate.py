import argparse, logging, math, os, sys, time, yaml, torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2

# Qiskit suite imports
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeMiami

from modules.data_pipeline.datasets import SwapDataset
from modules.utils.quantum_ops import tn2qiskit, amplitude_encoding
from modules.utils.tensor_ops import einsum2interleaved
from modules.utils.general import load_pkl, get_device
from modules.compilation.quantum.gates import Rx, Ry, Rz, CRz
from factory import build_dataset, build_experiment

# Framework path setup
ROOT_PATH = Path.cwd()
sys.path.insert(0, str(ROOT_PATH))

def run_circuits(qc_array, sampler, shots):
    job = sampler.run(qc_array, shots=shots)
    result_array = []

    primitive_result = job.result()
    for job in primitive_result:
        counts = job.join_data().get_counts()
        meas_qubits = len(next(iter(counts.items()))[0])

        post_selected_shots = counts.get("0" * (meas_qubits), 1)
        try:
            failed_state_counts = counts.get(f'1{"0" * (meas_qubits - 1)}', 1)
            ratio = post_selected_shots / (post_selected_shots + failed_state_counts)
            result_array.append(ratio)
        except:
            result_array.append(0.0)

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
    if config.get("backend") == "IBM":
        backend = FakeMiami()
        print(" Target Backend: FakeMiami (IBM Noise Blueprint Model)")
    else:
        backend = AerSimulator()
        print(" Target Backend: AerSimulator (Ideal State-Vector)")

    backend.set_options(max_parallel_threads=0, max_parallel_experiments=0)
    sampler = Sampler(backend)
    shots = config.get("shots", 4096)
    print(f" Evaluation Resolution: {shots} shots per circuit execution context")

    # 3. Dynamic Symbol Map Generation for Qiskit Translation
    log_phase("Extracting Parameter Angle Maps")
    txt_params_dict = {}
    if hasattr(text_model, "sym2param") and hasattr(text_model, "params"):
        for symbol, idx in text_model.sym2param.items():
            txt_params_dict[symbol] = float(text_model.params[idx].detach().cpu().item())
            
    img_params_dict = {}
    use_ansatz = not config.get("use_clip", True)
    if use_ansatz and hasattr(image_model, "sym2param") and hasattr(image_model, "params"):
        for symbol, idx in image_model.sym2param.items():
            img_params_dict[symbol] = float(image_model.params[idx].detach().cpu().item())

    print(f" Mapped parameters directly from live state: {len(txt_params_dict)} Text | {len(img_params_dict)} Vision.")

    log_phase("Constructing Data Engine Context")
    if config['vision'].get('use_clip', False):
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
            caption = sample["caption"]
            pos_img = sample["pos_img"]
            neg_img = sample["neg_img"]

            # 1. Generate Text Subcircuit Layout
            qc_txt, output_qubits, params_dict = tn2qiskit(
                einsum2interleaved(caption[0]), 
                caption[1], 
                meas_output=False, 
                all_params_dict=txt_params_dict
            )

            for q_idx in range(qc_txt.num_qubits):
                if q_idx not in output_qubits:
                    qc_txt.measure(q_idx, q_idx)

            if use_ansatz:
                pos_params = img_params_dict | image_model.encode_features(pos_img)
                qc_pos_img, _, _ = tn2qiskit(einsum2interleaved(image_model.einsum_expr), image_model.gate_arr, meas_output=False, all_params_dict=pos_params)
                neg_params = img_params_dict | image_model.encode_features(neg_img)
                qc_neg_img, _, _ = tn2qiskit(einsum2interleaved(image_model.einsum_expr), image_model.gate_arr, meas_output=False, all_params_dict=neg_params)
            else:
                pos_vector = pos_img.cpu().numpy() if hasattr(pos_img, "cpu") else np.array(pos_img)
                neg_vector = neg_img.cpu().numpy() if hasattr(neg_img, "cpu") else np.array(neg_img)
                
                normalized_pos = amplitude_encoding(pos_vector)
                normalized_neg = amplitude_encoding(neg_vector)
                
                required_qubits = int(math.ceil(math.log2(len(normalized_pos))))
                
                qc_pos_img = QuantumCircuit(required_qubits, 0)
                qc_pos_img.initialize(normalized_pos)
                
                qc_neg_img = QuantumCircuit(required_qubits, 0)
                qc_neg_img.initialize(normalized_neg)

            # 3. Assemble Positive Swap Test Frame
            qc_pos = qc_txt.copy()
            qreg_txt = qc_pos.qregs[0]
            qreg_img = qc_pos_img.qregs[0]
            qreg_anc_pos = QuantumRegister(1, "q_anc")
            creg_anc_pos = ClassicalRegister(1, "c_anc")

            qc_pos.add_register(qreg_img, qreg_anc_pos, creg_anc_pos)
            qc_pos.compose(qc_pos_img, qreg_img, inplace=True)

            qc_pos.h(qreg_anc_pos)
            for m_idx in range(len(output_qubits)):
                qc_pos.cswap(qreg_anc_pos, qreg_txt[output_qubits[m_idx]], qreg_img[m_idx])
            qc_pos.h(qreg_anc_pos)
            qc_pos.measure(qreg_anc_pos, creg_anc_pos)

            # 4. Assemble Negative Swap Test Frame
            qc_neg = qc_txt.copy()
            qreg_txt_neg = qc_neg.qregs[0]
            qreg_img_neg = qc_neg_img.qregs[0]
            qreg_anc_neg = QuantumRegister(1, "q_anc")
            creg_anc_neg = ClassicalRegister(1, "c_anc")

            qc_neg.add_register(qreg_img_neg, qreg_anc_neg, creg_anc_neg)
            qc_neg.compose(qc_neg_img, qreg_img_neg, inplace=True)
            
            qc_neg.h(qreg_anc_neg)
            for m_idx in range(len(output_qubits)):
                qc_neg.cswap(qreg_anc_neg, qreg_txt_neg[output_qubits[m_idx]], qreg_img_neg[m_idx])
            qc_neg.h(qreg_anc_neg)
            qc_neg.measure(qreg_anc_neg, creg_anc_neg)

            # Transpile directly for backend specifications
            pos_circs.append(transpile(qc_pos, backend))
            neg_circs.append(transpile(qc_neg, backend))

        except Exception as e:
            failed_circuits += 1
            tqdm.write(f" Circuit compilation dropped at sample index {idx}: {e}")

    if not pos_circs:
        print(" Terminal Execution Halt: No valid quantum circuits were assembled.")
        sys.exit(1)
    
    log_phase("Executing Emulator Engine Pipeline")
    print(f" Simulating quantum states across {len(pos_circs)} pairs...")

    start_eval_time = time.time()
    with torch.no_grad():
        pos_f = run_circuits(pos_circs, sampler, shots)
        neg_f = run_circuits(neg_circs, sampler, shots)
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