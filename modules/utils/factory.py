# factory.py
from modules.compilation.quantum.ansatz import CustomV5Ansatz, CustomV6Ansatz
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.compilation.classical.neural import MLPCompiler

from modules.models.text.einsum_quantum import VQCModel
from modules.models.text.einsum_classical import TNModel
from modules.models.text.neural_model import MLPModel

from modules.models.vision.quantum_map import QuantumFeatureMap, QFMap_CPTP
from modules.models.vision.clip import FrozenCLIP
from modules.models.vision.image_model import TTNImageModel

from modules.models.fusion.criteria import FS_InfoNCE, InfoNCE, UJ_InfoNCE
import importlib

def load_obj(import_str: str):
    module_path, attr_name = import_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)

def build_dataset(config: dict):
    ds_config = config['dataset']
    
    dataset_class = load_obj(ds_config['class'])
    collate_fn = load_obj(ds_config['collate_fn'])
    eval_mapper = load_obj(ds_config['eval_mapper'])
    
    return dataset_class, collate_fn, eval_mapper

def build_experiment(config, device):
    model_type = config['model_type']
    
    if model_type == 'tn':
        compiler = config['text']
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_dim']}
        mps_proc = MPS(bond_dim=config['text']['bond_dim'], max_order=config['text']['max_order'])
        ansatz = TNCompiler(obmap=obmap, decomp_fn=mps_proc)
        
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP().to(device)
        else:
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)

        text_model = TNModel(out_dim=config['embedding_dim']).to(device)
        loss_fn = InfoNCE()
        
    elif model_type == "vqc":
        compiler = config['text']
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_qubits']}
        ansatz = CustomV5Ansatz(obmap=obmap, layers=compiler['layers'])
        
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP(classical=False).to(device)
        else:
            image_model = QuantumFeatureMap(k=config['embedding_qubits'], 
                                            layers=config['vision']['layers'], 
                                            batch_size=config['batch_size'], 
                                            out_dim=config['out_dim'],
                                            id_init=False, 
                                            discard=config['discard'],
                                            method=config['vision']['method']).to(device)
            if config['discard']:
                image_model = QFMap_CPTP(base_image_model=image_model)

        text_model = VQCModel(out_q=config['embedding_qubits'], discard=config['discard']).to(device)
        if config['discard']:
            loss_fn = UJ_InfoNCE(label_smoothing=0.1)
        else:
            loss_fn = FS_InfoNCE(label_smoothing=0.1)
        
    elif model_type == "mlp":
        ansatz = MLPCompiler()
        if config['vision']['method'] == 'amp':
            image_model = FrozenCLIP().to(device)
        else:
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)
        text_model = MLPModel(hdim=config['text']['hidden_dim'], out_dim=config['embedding_dim']).to(device)
        loss_fn = InfoNCE()
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return ansatz, image_model, text_model, loss_fn