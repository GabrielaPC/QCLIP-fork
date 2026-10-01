import math, time, torch, mlflow, tempfile, pickle, os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMiami
from qiskit.transpiler import CouplingMap
# from qiskit.qiskit_iqm import IQMProvider

from modules.utils.tensor_ops import einsum2interleaved
from modules.utils.quantum_ops import tn2qiskit, amplitude_encoding
from modules.utils.general import gen_id, store_pkl

class BackendManager:
    def __init__(self, config, qlimit=None, hardware_profile=FakeMiami()):
        self.config = config
        self.amplitude_encode = True if config['vision']['method'] == 'amp' else False
        self.discard = self.config.get('discard', False)
        self.meas_method = self.config.get('emulate', {}).get('measurement_method', 'compute_uncompute')
        self.out_q = self.config.get('embedding_qubits', None) if self.discard else None
        self.qlimit = qlimit
        self._backend = None
        self.coupling_map = None
        self.basis_gates = None
        self.hardware_profile = hardware_profile
        self._setup_backend()

    def _setup_backend(self):
        # if self.config.get('emulate', {}).get('backend') == 'iqm':
        #     token_var = self.cfg.get('tokens_env_var', 'IQM_TOKEN')
        #     api_token = os.environ.get(token_var)
        #     server_url = self.cfg.get('server_url', 'https://cocos.resonance.meetiqm.com/garnet')

        devices = AerSimulator().available_devices()
        qdev = 'GPU' if 'GPU' in devices else 'CPU'
        method = self.config['emulate'].get('backend', 'statevector')
        
        backend_options = {
            "device": qdev,
            "method": method,
            "precision": "double",
            "max_parallel_experiments": 0,
            "max_parallel_threads": 0,
            "max_parallel_shots": 0,
            "fusion_enable": True,
            "fusion_threshold": 2,
            "fusion_max_qubit": 6,
            # "seed_simulator": 42,
            # cuStateVec_enable=True
        }

        edges = self.hardware_profile.coupling_map.get_edges()
        if self.qlimit is not None:
            filtered_edges = [edge for edge in edges if edge[0] < self.qlimit and edge[1] < self.qlimit]
            self.coupling_map = CouplingMap(filtered_edges)
        else:
            self.coupling_map = self.hardware_profile.coupling_map
        self.basis_gates = self.hardware_profile.basis_gates

        if self.config['emulate'].get('noise'):
            noise_model = NoiseModel.from_backend(self.hardware_profile)
            backend_options["noise_model"] = noise_model 
            backend_options["seed_simulator"] = 42
            backend_options['shot_branching_enable'] = True
            backend_options["statevector_parallel_threshold"] = 0
            backend_options["batched_shots_gpu_max_qubits"] = 30
            if method == 'statevector':
                backend_options["batched_shots_gpu"] = True
                backend_options["max_parallel_shots"] = 0
        else:
            if method == 'statevector':
                backend_options["statevector_sample_measure_opt"] = 10
                backend_options["batched_shots_gpu"] = True
                backend_options["max_parallel_shots"] = 256
        if method == 'matrix_product_state':
            backend_options["matrix_product_state_max_bond_dimension"] = 128
            backend_options["matrix_product_state_truncation_threshold"] = 1e-8
            backend_options["matrix_product_state_parallel_swap"] = True

        print(f" Simulation: Profile {self.hardware_profile.name} ({self.qlimit} qubits), Device {qdev}, Method {method}, Noise {'Enabled' if self.config['emulate'].get('noise') else 'Disabled'}")
        if method == 'tensor_network':
            self._backend = AerSimulator(method=method, device=qdev)
        else:
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
                # layout_method='trivial',
                # seed_transpiler=42,
                num_processes=4  # Kept strict to manage RAM limits
            )
            transpiled_circs.extend(transpiled_chunk)
        return transpiled_circs
    
    def compile_circuits(self, dataset, text_model, image_model, limit=None, opt_lvl=1):
        data_size = min(limit, len(dataset)) if limit else len(dataset)
        print(f" Compiling {data_size} circuit pairs from dataset with {len(dataset)} samples.")

        txt_params = text_model._get_params()
        img_params = image_model._get_params()             

        raw_pos_circs, raw_neg_circs = [], []
        failed_circuits = 0

        for idx in tqdm(range(data_size), desc="Compiling Qiskit DAGs", unit="pair"):
            try:
                sample = dataset[idx]
                image = sample["image"]
                pos_einsum, pos_wires = sample["pos_caption"]
                neg_einsum, neg_wires = sample["neg_caption"]

                qc_pos_txt, pos_out_q, pos_txt_p = tn2qiskit(
                    einsum2interleaved(pos_einsum), pos_wires, txt_params, out_q=self.out_q, meas_output=False
                )
                qc_neg_txt, neg_out_q, neg_txt_p = tn2qiskit(
                    einsum2interleaved(neg_einsum), neg_wires, txt_params, out_q=self.out_q, meas_output=False
                )

                if qc_pos_txt.num_qubits > self.qlimit or qc_neg_txt.num_qubits > self.qlimit:
                    failed_circuits += 1
                    continue

                if self.amplitude_encode:
                    img_vec = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
                    normed_img_vec = amplitude_encoding(img_vec)
                    qc_img = QuantumCircuit(int(math.ceil(math.log2(len(normed_img_vec)))), 0)
                    qc_img.initialize(normed_img_vec)
                    img_p = {}
                else:
                    img_vars = img_params | image_model.encode_features(image)
                    in_idx, out_idx = einsum2interleaved(image_model.einsum_expr.replace('b', ''))
                    qc_img, _, img_p = tn2qiskit([in_idx, out_idx], image_model.gate_arr, img_vars, out_q=self.out_q, meas_output=False)

                pos_txt_p = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in pos_txt_p.items()}
                neg_txt_p = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in neg_txt_p.items()}
                img_p = {k: float(v.item()) if hasattr(v, 'item') else float(v) for k, v in img_p.items()}

                qc_pos_txt.assign_parameters(pos_txt_p, inplace=True)
                qc_neg_txt.assign_parameters(neg_txt_p, inplace=True)
                if img_p: qc_img.assign_parameters(img_p, inplace=True)

                # Assemble frames
                if self.meas_method == 'compute_uncompute':
                    qc_pos = self._compute_uncompute(qc_pos_txt, qc_img, pos_out_q)
                    qc_neg = self._compute_uncompute(qc_neg_txt, qc_img, neg_out_q)
                elif self.meas_method == 'destructive_swap':
                    qc_pos = self._destructive_swap(qc_pos_txt, qc_img, pos_out_q)
                    qc_neg = self._destructive_swap(qc_neg_txt, qc_img, neg_out_q)

                qc_pos.metadata = {"output_qubits": pos_out_q}
                qc_neg.metadata = {"output_qubits": neg_out_q}
                raw_pos_circs.append(qc_pos)
                raw_neg_circs.append(qc_neg)
                
            except Exception as e:
                if isinstance(e, KeyError):
                    print(f"\n[DEBUG] KeyError on index {idx}. Available keys in sample: {list(sample.keys())}")
                print(f" Error compiling circuit pair at index {idx}: {e}")
                failed_circuits += 1
        
        print(f" Compiled {len(raw_pos_circs)} circuit pairs | Dropped {failed_circuits} samples.")
        pos_circs = self.transpile(raw_pos_circs, batch_size=256, optimization_level=opt_lvl)
        neg_circs = self.transpile(raw_neg_circs, batch_size=256, optimization_level=opt_lvl)

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
            qc.compose(qc_img.inverse(), qubits=out_q, inplace=True)
            qc.measure(out_q, range(n_out))
        else:
            qc = QuantumCircuit(n_txt, n_txt)
            qc.compose(qc_txt, inplace=True)
            qc.compose(qc_img.inverse(), qubits=out_q, inplace=True)
            qc.measure(range(n_txt), range(n_txt))
        return qc
    
    def _destructive_swap(self, qc_txt, qc_img, out_q):
        qc_txt = self._strip_clbits(qc_txt)
        qc_img = self._strip_clbits(qc_img)

        n_out = len(out_q)
        n_txt = qc_txt.num_qubits
        n_total = n_txt + n_out

        if self.discard:
            qc = QuantumCircuit(n_total, 2 * n_out)
            qc.compose(qc_txt, range(n_txt), inplace=True)
            qc.compose(qc_img, range(n_txt, n_total), inplace=True)

            for i, q_out in enumerate(out_q):
                img_q = n_txt + i 
                qc.cx(q_out, img_q)
                qc.h(q_out)
                qc.measure(q_out, i)
                qc.measure(img_q, n_out + i)
        else:
            qc = QuantumCircuit(n_total, n_total)
            qc.compose(qc_txt, range(n_txt), inplace=True)
            qc.compose(qc_img, range(n_txt, n_total), inplace=True)

            for i, q_out in enumerate(out_q):
                img_q = n_txt + i 
                qc.cx(q_out, img_q)
                qc.h(q_out)
            qc.measure(range(n_total), range(n_total))
        return qc

class Emulator:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.shots = 64
        self.discard = self.config.get('discard', False)
        self.meas_method = self.config.get('emulate', {}).get('measurement_method', 'compute_uncompute')
        self.dataset_name = config['dataset']['name']
        self.experiment_name = gen_id(config)

    def shot_estimation(self, nq_out, nq_ps=0, resolution=50):
        raw_shots = resolution * max(nq_ps, 1) * (2 ** nq_out)
        self.shots = min(int(math.ceil(raw_shots)), 1_000_000)

    def run_circuit(self, qc_array, shots=None, batch_size=24):
        result_array = []
        chunks = [qc_array[i:i + batch_size] for i in range(0, len(qc_array), batch_size)]

        for chunk in tqdm(chunks, desc="Running Circuits"):
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
            # print(f" Batch of {len(chunk)} circuits executed. Current fidelity results: {np.average(result_array[-len(chunk):]):.4f}")
        return torch.tensor(result_array, dtype=torch.float32)
    
    def _eval_compute_uncompute(self, qc, counts):
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        out_qs = set(qc.metadata.get("output_qubits", []))
        num_clbits = qc.num_clbits

        if self.discard:
            target_str = "0" * num_clbits
            return counts.get(target_str, 0) / total_shots
        else:
            accepted_shots = 0
            all_zero_shots = counts.get("0" * num_clbits, 0)
            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]  
                if all(rev_bstr[q] == '0' for q in range(num_clbits) if q not in out_qs):
                    accepted_shots += count
            if accepted_shots == 0:
                return 0.0
            return all_zero_shots / accepted_shots
        
    def _eval_destructive_swap(self, qc, counts):
        out_qs = qc.metadata.get("output_qubits", [])
        n_out = len(out_qs)

        accepted_shots = 0
        weighted_parity_sum = 0

        if self.discard:
            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]
                k = sum(
                    int(rev_bstr[i]) * int(rev_bstr[n_out + i]) 
                    for i in range(n_out)
                )
                parity_sign = 1 if k % 2 == 0 else -1
                weighted_parity_sum += count * parity_sign
                accepted_shots += count
        else:
            n_txt = qc.num_clbits - n_out
            aux_qs = [q for q in range(n_txt) if q not in out_qs]

            for bstr, count in counts.items():
                rev_bstr = bstr[::-1]
                if all(rev_bstr[q] == '0' for q in aux_qs):
                    k = sum(
                        int(rev_bstr[out_qs[i]]) * int(rev_bstr[n_txt + i])
                        for i in range(n_out)
                    )
                    parity_sign = 1 if k % 2 == 0 else -1
                    weighted_parity_sum += count * parity_sign
                    accepted_shots += count
        if accepted_shots == 0:
            return 0.0

        fidelity = weighted_parity_sum / accepted_shots
        return max(0.0, float(fidelity))

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
                "embedding_qubits": self.config.get('embedding_qubits', 0),
                "discard_qubits": self.config.get('discard', False),
                "epsilon": self.config.get('emulate', {}).get('eps', 0.01),
                "shots": self.shots,
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

    def run_experiment(self, pos_circs, neg_circs, batch_size=24, shots=None, num_runs=5):
        shots = shots if shots is not None else self.shots
        torch.cuda.empty_cache()

        all_accuracies = []
        all_pos_fidelities = []
        all_neg_fidelities = []

        base_seed = int(np.random.randint(0, 2**31 - 1))

        for run_idx in range(num_runs):
            self.backend.set_options(seed_simulator=base_seed + run_idx) 

            start_eval_time = time.time()
            pos_raw = self.run_circuit(pos_circs, shots=shots, batch_size=batch_size)
            neg_raw = self.run_circuit(neg_circs, shots=shots, batch_size=batch_size)
            elapsed = time.time() - start_eval_time

            correct = (pos_raw > neg_raw).float()
            accuracy = correct.mean().item()

            all_accuracies.append(accuracy)
            all_pos_fidelities.append(pos_raw.mean().item())
            all_neg_fidelities.append(neg_raw.mean().item())

            print(f" Trial {run_idx + 1} ({elapsed:.2f}s) | Accuracy: {accuracy * 100:.2f}% | Mean Pos Fid: {pos_raw.mean().item():.4f} | Mean Neg Fid: {neg_raw.mean().item():.4f}")

        accs = np.array(all_accuracies)
        pos_fids = np.array(all_pos_fidelities)
        neg_fids = np.array(all_neg_fidelities)

        n = len(accs)
        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs, ddof=1)) if n > 1 else 0.0
        acc_var = float(np.var(accs, ddof=1)) if n > 1 else 0.0
        acc_ci95 = float(1.96 * (acc_std / np.sqrt(n))) if n > 1 else 0.0

        pos_mean = float(np.mean(pos_fids))
        pos_std = float(np.std(pos_fids, ddof=1)) if n > 1 else 0.0
        neg_mean = float(np.mean(neg_fids))
        neg_std = float(np.std(neg_fids, ddof=1)) if n > 1 else 0.0

        # if self.discard:
        #     all_raw = torch.cat([pos_raw, neg_raw])
        #     f_min, f_max = all_raw.min(), all_raw.max()
        #     if f_max - f_min >= 1e-6:
        #         pos_raw = (pos_raw - f_min) / (f_max - f_min)
        #         neg_raw = (neg_raw - f_min) / (f_max - f_min)
        # pos_f = torch.asin((0.5 + 0.5 * pos_raw).abs().clamp(0, 1))
        # neg_f = torch.asin((0.5 + 0.5 * neg_raw).abs().clamp(0, 1))

        return {
            "accuracy": acc_mean,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "accuracy_var": acc_var,
            "accuracy_ci95": acc_ci95,
            "pos_fidelity_mean": pos_mean,
            "pos_fidelity_std": pos_std,
            "neg_fidelity_mean": neg_mean,
            "neg_fidelity_std": neg_std,
            "num_runs": num_runs,
            "raw_accuracies": all_accuracies,
        }