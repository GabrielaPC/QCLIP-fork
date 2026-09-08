import math, time, torch, mlflow, tempfile, pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMiami
from qiskit.transpiler import CouplingMap

from modules.utils.tensor_ops import einsum2interleaved
from modules.utils.quantum_ops import tn2qiskit, amplitude_encoding
from modules.utils.general import gen_id, store_pkl

class BackendManager:
    def __init__(self, config, qlimit=None, hardware_profile=FakeMiami()):
        self.config = config
        self.amplitude_encode = True if config['vision']['method'] == 'amp' else False
        self.discard = self.config.get('emulate', {}).get('discard', False)
        self.meas_method = self.config.get('emulate', {}).get('measurement_method', 'compute_uncompute')
        self.nshadows = 0
        self.qlimit = qlimit
        self._backend = None
        self.coupling_map = None
        self.basis_gates = None
        self.hardware_profile = hardware_profile
        self._setup_backend()

    def _setup_backend(self):
        devices = AerSimulator().available_devices()
        qdev = 'GPU' if 'GPU' in devices else 'CPU'
        method = self.config['emulate'].get('backend', 'statevector')
        
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
            # cuStateVec_enable=True
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

                # Assemble frames
                if self.meas_method == 'compute_uncompute':
                    qc_pos = self._compute_uncompute(qc_pos_txt, qc_img, pos_out_q)
                    qc_neg = self._compute_uncompute(qc_neg_txt, qc_img, neg_out_q)
                elif self.meas_method == 'destructive_swap':
                    qc_pos = self._destructive_swap(qc_pos_txt, qc_img, pos_out_q)
                    qc_neg = self._destructive_swap(qc_neg_txt, qc_img, neg_out_q)
                elif self.meas_method == 'classical_shadows':
                    qc_pos = self._classical_shadows(qc_pos_txt, qc_img, pos_out_q)
                    qc_neg = self._classical_shadows(qc_neg_txt, qc_img, neg_out_q)

                pos_metadata.append({"output_qubits": pos_out_q})
                neg_metadata.append({"output_qubits": neg_out_q})
                raw_pos_circs.append(qc_pos)
                raw_neg_circs.append(qc_neg)
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
    

    def _strip_clbits(self, qc):
        if qc.num_clbits == 0:
            return qc
        clean_qc = QuantumCircuit(qc.num_qubits)
        for instr in qc.data:
            if instr.operation.name != 'measure':
                qargs = [qc.find_bit(q).index for q in instr.qubits]
                clean_qc.append(instr.operation, qargs)
        return clean_qc

    def _compute_uncompute(self, qc_txt, qc_img, out_q):
        qc_txt = self._strip_clbits(qc_txt)
        qc_img = self._strip_clbits(qc_img)

        n_txt = qc_txt.num_qubits
        n_out = len(out_q)

        if self.discard:
            qc = QuantumCircuit(n_txt, n_out)
            qc.compose(qc_txt, inplace=True)
            qc.compose(qc_img.inverse(), qubits=out_q[::-1], inplace=True)
            qc.measure(out_q, range(n_out))
        else:
            qc = QuantumCircuit(n_txt, n_txt)
            qc.compose(qc_txt, inplace=True)
            qc.compose(qc_img.inverse(), qubits=out_q[::-1], inplace=True)
            qc.measure(range(n_txt), range(n_txt))
        return qc
    
    def _destructive_swap(self, qc_txt, qc_img, out_q):
        qc_txt = self._strip_clbits(qc_txt)
        qc_img = self._strip_clbits(qc_img)

        n_out = len(out_q)
        n_txt = qc_txt.num_qubits
        n_total = n_txt + n_out

        qc = QuantumCircuit(n_total, 2 * n_out if self.discard else n_total)
        qc.compose(qc_txt, range(n_txt), inplace=True)
        qc.compose(qc_img, range(n_txt, n_total), inplace=True)

        for i, q_out in enumerate(out_q):
            img_q = n_txt + i 
            qc.cx(q_out, img_q)
            qc.h(q_out)
        
        if self.discard:
            qc.measure(out_q, range(n_out))
            qc.measure(range(n_txt, n_total), range(n_out, 2 * n_out))
        else:
            qc.measure(range(n_total), range(n_total))
        return qc
    
    def _classical_shadows(self, qc_txt, qc_img, out_q):
        qc_txt = self._strip_clbits(qc_txt)
        qc_img = self._strip_clbits(qc_img)

        n_q = qc_txt.num_qubits
        cbits = len(out_q) if self.discard else n_q
        qc = QuantumCircuit(n_q, cbits)
        qc.compose(qc_txt, inplace=True)
        qc.compose(qc_img.inverse(), qubits=out_q, inplace=True)
        return qc

class Emulator:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.shots = 64
        self.nshadows = 100
        self.discard = self.config.get('emulate', {}).get('discard', False)
        self.meas_method = self.config.get('emulate', {}).get('measurement_method', 'compute_uncompute')
        self.dataset_name = config['dataset']['name']
        self.experiment_name = gen_id(config)

    def shot_estimation(self, nq_out, nq_ps, epsilon=0.01):    
        if self.meas_method == "classical_shadows":
            self.shots = 1
            return
        req_shots = 0.25 / (epsilon ** 2)
        ps_factor = 1 if self.discard else (2 ** nq_ps)
        raw_shots = req_shots * ps_factor
        # min_shots = (2 ** nq_out) * 10
        # final_shots = max(raw_shots, min_shots)
        final_shots = int(math.ceil(raw_shots))
        self.shots = min(final_shots, 1_000_000) # max(1024, min(final_shots, 1_000_000))

    def shadow_estimation(self, nq_out, epsilon=0.01, confidence=0.95):
        delta = 1.0 - confidence
        shadow_norm = 3 ** nq_out
        
        raw_shadows = (shadow_norm * 2.0 * math.log(2.0 / delta)) / (epsilon ** 2)
        
        min_shadows = 10
        max_shadows = 10_000
        
        optimal_shadows = max(min_shadows, min(int(math.ceil(raw_shadows)), max_shadows))
        self.nshadows = optimal_shadows

    def run_circuit(self, qc_array, shots=None, batch_size=24):
        shots = shots if shots is not None else self.shots
        result_array = []
        chunks = [qc_array[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]

        for chunk in tqdm(chunks, desc="Running Circuits"):
            if self.meas_method == "classical_shadows":
                exec_chunk, shadow_metas = self.process_shadow_chunk(chunk)
                job = self.backend.run(exec_chunk, shots=shots)
                result = job.result()
                for i, meta in enumerate(shadow_metas):
                    offset = i * self.nshadows
                    sample_counts = [result.get_counts(offset + s) for s in range(self.nshadows)]
                    fidelity = self._eval_classical_shadows(meta["bases"], meta["nout"], meta["num_clbits"], sample_counts)
                    result_array.append(fidelity)
            else:
                job = self.backend.run(chunk, shots=shots)
                result = job.result()

                for i, qc in enumerate(chunk):
                    counts = result.get_counts(i)
                    
                    if self.meas_method == "compute_uncompute":
                        fidelity = self._eval_compute_uncompute(qc, counts)
                    elif self.meas_method == "destructive_swap":
                        fidelity = self._eval_destructive_swap(qc, counts)
                    else:
                        raise ValueError(f"Unknown measurement method: {self.meas_method}")
                    result_array.append(fidelity)
        
        return torch.tensor(result_array, dtype=torch.float32)
    
    def _eval_compute_uncompute(self, qc, counts):
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        out_qs = set(qc.metadata.get("output_qubits", []))
        num_out = len(out_qs) if out_qs else qc.num_clbits

        if self.discard:
            target_str = "0" * num_out
            return counts.get(target_str, 0) / total_shots
        else:
            success_shots = 0
            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]  # Align string indices with qubit indices
                if all(rev_bstr[q] == '0' for q in out_qs):
                    success_shots += count

            return success_shots / total_shots
        
    def _eval_destructive_swap(self, qc, counts):
        n_out = len(qc.metadata.get("output_qubits", range(qc.num_clbits)))
        num_clbits = qc.num_clbits

        accepted_shots = sum(counts.values())
        weighted_parity_sum = 0
        
        for bstr, count in counts.items():
            txt_bits = [int(bstr[num_clbits - 1 - i]) for i in range(n_out)]
            img_bits = [int(bstr[num_clbits - 1 - (n_out + i)]) for i in range(n_out)]
            parity = sum(a * b for a, b in zip(txt_bits, img_bits)) % 2
            weighted_parity_sum += count * (1 if parity == 0 else -1)
        return max(0.0, weighted_parity_sum / accepted_shots) if accepted_shots > 0 else 0.0
    
    def process_shadow_chunk(self, chunk):
        exec_chunk = []
        shadow_metas = []

        for qc in chunk:
            out_q = qc.metadata["output_qubits"]
            n_out = len(out_q)
            cbits = n_out if self.discard else qc.num_qubits
            bases_list = []

            for _ in range(self.nshadows):
                qc_shadow = qc.copy()
                bases = np.random.choice([0, 1, 2], size=n_out)  # 0=Z, 1=X, 2=Y
                bases_list.append(bases)

                for i, q in enumerate(out_q):
                    if bases[i] == 1: qc_shadow.ry(-np.pi / 2, q) # X basis: RY(-pi/2)
                    elif bases[i] == 2: qc_shadow.rx(np.pi / 2, q) # Y basis: RX(pi/2)

                if self.discard: qc_shadow.measure(out_q, range(n_out))
                else: qc_shadow.measure(range(qc.num_qubits), range(cbits))
                exec_chunk.append(qc_shadow)

            shadow_metas.append({
                "bases": bases_list,
                "nout": n_out,
                "num_clbits": cbits
            })

        return exec_chunk, shadow_metas

    def _eval_classical_shadows(self, bases_list, nout, num_clbits, counts_list):
        shadow_estimates = []
        for bases, counts in zip(bases_list, counts_list):
            total_shots = sum(counts.values())
            if total_shots == 0:
                shadow_estimates.append(0.0)
                continue

            exp_val = 0.0
            for bstr, count in counts.items():
                bits = [int(bstr[num_clbits - 1 - i]) for i in range(nout)]
                val = 1.0
                for bit, b in zip(bits, bases):
                    val *= (2.0 if bit == 0 else -1.0) if b == 0 else 0.5
                exp_val += val * (count / total_shots)

            shadow_estimates.append(exp_val)

        return float(np.mean(shadow_estimates))

    def log_experiment(self, data, exec_time, einsum_data):
        mlf_db_path = Path.cwd() / f"mlf_dbs/{self.dataset_name}.db"
        mlf_db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.pytorch.autolog(log_models=False)
        mlflow.set_tracking_uri(f"sqlite:///{mlf_db_path}")
        mlflow.set_experiment(self.dataset_name+'emul')

        meas_method = self.config.get('emulate', {}).get('measurement_method', 'compute_uncompute')
        with mlflow.start_run(run_name=f"{self.experiment_name}"):
            mlflow.log_params({
                "dataset": self.dataset_name,
                "measurement_method": meas_method,
                "discard_qubits": self.config.get('emulate', {}).get('discard', False),
                "epsilon": self.config.get('emulate', {}).get('eps', 0.01),
                "shots": self.shots,
                "nshadows": self.nshadows,
                "backend_method": self.config.get('emulate', {}).get('backend', 'statevector'),
                "noise_enabled": self.config.get("noise", False), 
                "vision_method": self.config.get('vision', {}).get('method'),
            })

            if einsum_data:
                max_dict = {'max_qubits': einsum_data['max'][0], 
                            'max_gates': einsum_data['max'][1], 
                            'max_depth': einsum_data['max'][2], 
                            'max_intermediate_state': einsum_data['max'][3], 
                            'max_2q_gates': einsum_data['max'][4]}
                mlflow.log_metrics(max_dict)
                avg_dict = {'avg_qubits': einsum_data['avg'][0], 
                            'avg_gates': einsum_data['avg'][1], 
                            'avg_depth': einsum_data['avg'][2], 
                            'avg_intermediate_state': einsum_data['avg'][3], 
                            'avg_2q_gates': einsum_data['avg'][4]}
                mlflow.log_metrics(avg_dict)
            
            mlflow.log_metrics({'execution_time': exec_time, 'accuracy': data['accuracy']})
            with tempfile.TemporaryDirectory() as tmp_dir:
                raw_pkl_path = Path(tmp_dir) / "raw_data.pkl"
                with open(raw_pkl_path, "wb") as f:
                    pickle.dump(data, f)
                mlflow.log_artifact(str(raw_pkl_path), artifact_path="raw_outputs")
        print(f" Logged run metrics & artifacts to: {mlf_db_path}")

    def run_experiment(self, pos_circs, neg_circs, batch_size=24, einsum_data=None):
        start_eval_time = time.time()
        pos_raw = self.run_circuit(pos_circs, batch_size)
        neg_raw = self.run_circuit(neg_circs, batch_size)

        if self.discard:
            all_raw = torch.cat([pos_raw, neg_raw])
            f_min, f_max = all_raw.min(), all_raw.max()
            if f_max - f_min >= 1e-6:
                pos_raw = (pos_raw - f_min) / (f_max - f_min)
                neg_raw = (neg_raw - f_min) / (f_max - f_min)
        pos_f = torch.asin((0.5 + 0.5 * pos_raw).abs().clamp(0, 1))
        neg_f = torch.asin((0.5 + 0.5 * neg_raw).abs().clamp(0, 1))

        elapsed = time.time() - start_eval_time
        print(f"    Completed in {elapsed:.2f}s")

        pos_arr = pos_f.detach().cpu().numpy().flatten()
        neg_arr = neg_f.detach().cpu().numpy().flatten()
        margins = pos_arr - neg_arr
        correct = margins > 0
        accuracy = np.mean(correct)

        data = {"accuracy": accuracy, "pos_scores": pos_arr, "neg_scores": neg_arr, "margin": margins, "correct": correct}
        self.log_experiment(data, elapsed, einsum_data)
        return data