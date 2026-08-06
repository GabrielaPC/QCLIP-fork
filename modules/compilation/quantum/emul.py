import math
import numpy as np
import time
import torch
import os
from tqdm import tqdm
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMiami
from qiskit.transpiler import CouplingMap
from modules.utils.tensor_ops import einsum2interleaved
from modules.utils.quantum_ops import tn2qiskit, amplitude_encoding
from modules.utils.general import gen_id
from modules.utils.general import store_pkl

class BackendManager:
    def __init__(self, config, qlimit=None, hardware_profile=FakeMiami()):
        self.config = config
        self.amplitude_encode = True if config['vision']['method'] == 'amp' else False
        self.qlimit = qlimit
        self._backend = None
        self.coupling_map = None
        self.basis_gates = None
        self.hardware_profile = hardware_profile
        self._setup_backend()

    def _setup_backend(self):
        devices = AerSimulator().available_devices()
        qdev = 'GPU' if 'GPU' in devices else 'CPU'
        method = self.config['emulate'].get('method', 'statevector')
        
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

        edges = self.hardware_profile.coupling_map.get_edges()
        
        if self.qlimit is not None:
            filtered_edges = [edge for edge in edges if edge[0] < self.qlimit and edge[1] < self.qlimit]
            self.coupling_map = CouplingMap(filtered_edges)
        else:
            self.coupling_map = self.hardware_profile.coupling_map

        self.basis_gates = self.hardware_profile.basis_gates

        if self.config.get('noise'):
            noise_model = NoiseModel.from_backend(self.hardware_profile)
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

        print(f" Simulation: Profile {self.hardware_profile.name} ({self.qlimit} qubits), Device {qdev}, Method {method}, Noise {'Enabled' if self.config.get('noise') else 'Disabled'}")
        
        self._backend = AerSimulator(method=method, device=qdev, cuStateVec_enable=True) 
        self._backend.set_options(**backend_options)

    @property
    def backend(self):
        return self._backend

    @property
    def get_physical_circuit(self):
        return self.basis_gates, self.coupling_map
    
    def transpile(self, circuits, batch_size=128, optimization_level=1):
        transpiled_circs = []
        chunks = [circuits[i:i + batch_size] for i in range(0, len(circuits), batch_size)]
        
        for chunk in tqdm(chunks, desc="Transpiling", unit="batch"):
            transpiled_chunk = transpile(
                chunk, 
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates,
                optimization_level=optimization_level,
                num_processes=4  # Kept strict to manage RAM limits
            )
            transpiled_circs.extend(transpiled_chunk)
        return transpiled_circs
    
    def compile_circuits(self, dataset, text_model, image_model, limit=None):
        data_size = min(limit, len(dataset)) if limit else len(dataset)
        print(f" Compiling {data_size} circuit pairs from dataset with {len(dataset)} samples.")
        txt_params = text_model._get_params()
        img_params = image_model._get_params() 

        raw_pos_circs, raw_neg_circs = [], []
        pos_metadata, neg_metadata = [], []
        pos_param_maps, neg_param_maps = [], []
        failed_circuits = 0

        for idx in tqdm(range(data_size), desc="Compiling Qiskit DAGs", unit="pair"):
            try:
                sample = dataset[idx]
                image = sample["image"]
                pos_einsum, pos_wires = sample["pos_caption"]
                neg_einsum, neg_wires = sample["neg_caption"]

                qc_pos_txt, pos_out_q, pos_txt_p = tn2qiskit(
                    einsum2interleaved(pos_einsum), pos_wires, txt_params, False
                )
                qc_neg_txt, neg_out_q, neg_txt_p = tn2qiskit(
                    einsum2interleaved(neg_einsum), neg_wires, txt_params, False
                )

                if qc_pos_txt.num_qubits > self.qlimit or qc_neg_txt.num_qubits > self.qlimit:
                    failed_circuits += 1
                    continue

                if self.amplitude_encode:
                    img_vec = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
                    normed_img_vec = amplitude_encoding(img_vec)
                    qc_img = QuantumCircuit(int(math.ceil(math.log2(len(normed_img_vec)))), 0)
                    qc_img.initialize(normed_img_vec)
                    pos_params, neg_params = pos_txt_p, neg_txt_p
                else:
                    img_vars = img_params | image_model.encode_features(image)
                    in_idx, out_idx = einsum2interleaved(image_model.einsum_expr.replace('b', ''))
                    qc_img, _, img_p = tn2qiskit([in_idx, out_idx], image_model.gate_arr, img_vars, False)
                    pos_params = pos_txt_p | img_p
                    neg_params = neg_txt_p | img_p


                pos_params = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in pos_params.items()}
                neg_params = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in neg_params.items()}
                qc_img_inv = qc_img.inverse()

                # Assemble frames
                qc_pos = QuantumCircuit(qc_pos_txt.num_qubits, qc_pos_txt.num_clbits)
                qc_pos.compose(qc_pos_txt, inplace=True)
                qc_pos.compose(qc_img_inv, qubits=pos_out_q, inplace=True)
                qc_pos.measure(pos_out_q, pos_out_q)

                qc_neg = QuantumCircuit(qc_neg_txt.num_qubits, qc_neg_txt.num_clbits)
                qc_neg.compose(qc_neg_txt, inplace=True)
                qc_neg.compose(qc_img_inv, qubits=neg_out_q, inplace=True)
                qc_neg.measure(neg_out_q, neg_out_q)

                raw_pos_circs.append(qc_pos)
                raw_neg_circs.append(qc_neg)
                pos_metadata.append({"output_qubits": pos_out_q})
                neg_metadata.append({"output_qubits": neg_out_q})
                pos_param_maps.append(pos_params)
                neg_param_maps.append(neg_params)
                
            except Exception as e:
                if isinstance(e, KeyError):
                    print(f"\n[DEBUG] KeyError on index {idx}. Available keys in sample: {list(sample.keys())}")
                print(f" Error compiling circuit pair at index {idx}: {e}")
                failed_circuits += 1
        
        print(f" Compiled {len(raw_pos_circs)} circuit pairs | Dropped {failed_circuits} samples.")
        print(f"\n[CPU] Transpiling {len(raw_pos_circs) * 2} circuits using pool allocations...")
        pos_circs = self.transpile(raw_pos_circs, batch_size=256, optimization_level=2)
        neg_circs = self.transpile(raw_neg_circs, batch_size=256, optimization_level=2)

        for i in range(len(pos_circs)):
            pos_circs[i].metadata = pos_metadata[i]
            pos_circs[i].assign_parameters(pos_param_maps[i], inplace=True)
            
            neg_circs[i].metadata = neg_metadata[i]
            neg_circs[i].assign_parameters(neg_param_maps[i], inplace=True)

        return pos_circs, neg_circs

class Emulator:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.shots = 64

    def shot_estimation(self, nq_out, nq_ps, epsilon=0.01):    
        req_shots = 0.25 / (epsilon ** 2)
        raw_shots = req_shots * (2 ** nq_ps)
        min_shots = (2 ** nq_out) * 10
        final_shots = max(raw_shots, min_shots)
        final_shots = int(math.ceil(final_shots))
        self.shots = max(4096, min(final_shots, 1_000_000))

    def run_circuit(self, qc_array, shots=None, batch_size=24):
        shots = shots if shots is not None else self.shots
        result_array = []
        indices = range(len(qc_array))
        chunks = [indices[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]

        for idx_list in tqdm(chunks, desc="Running Circuits"):
            chunk_circs = [qc_array[i] for i in idx_list]
            job = self.backend.run(chunk_circs, shots=shots)
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
                    if all(bitstring[idx_bit] == '0' for idx_bit in check_indices):
                        accepted_shots += count
                        if bitstring == zero_str:
                            all_zero_shots += count

                fidelity = all_zero_shots / accepted_shots if accepted_shots > 0 else 0.0
                result_array.append(fidelity)
                
        result_tensor = 0.5 + (0.5 * torch.tensor(result_array))
        return torch.asin(result_tensor.abs().clamp(0, 1))

    def run_experiment(self, pos_circs, neg_circs, batch_size=24):
        start_eval_time = time.time()
        pos_f = self.run_circuit(pos_circs, batch_size)
        neg_f = self.run_circuit(neg_circs, batch_size)
        elapsed = time.time() - start_eval_time
        print(f"    Completed in {elapsed:.2f}s")

        run_type = "noisy" if self.config.get("noise") else "noiseless"
        run_name = gen_id(self.config)
        save_filename = f"results/{self.config['dataset']['name']}/{run_name}_{run_type}"
        if run_type == "noisy":
            save_filename += f"{self.config['eps']}"
        save_filename += "_margins.pkl"

        def to_ndarray(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().flatten()
            return np.asarray(x).flatten()

        pos_arr = to_ndarray(pos_f)
        neg_arr = to_ndarray(neg_f)
        margins = pos_arr - neg_arr
        correct = margins > 0

        data = {"pos_scores": pos_arr, "neg_scores": neg_arr, "margin": margins, "correct": correct}
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        store_pkl(data, save_filename)
        print(f" Saved emulation results to {save_filename}")
        return data