import torch, pickle, gc, random, logging, yaml, os
from datetime import datetime
import numpy as np
from pathlib import Path

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"\n[Warning] CUDA is physically available but incompatible or broken.")
            print(f"Error detail: {e}")
            print("Falling back to CPU execution for stability...\n")
    return torch.device("cpu")

def store_pkl(data, fpathname):
    with open(fpathname, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(fpathname):
    gc.disable()
    try:
        with open(fpathname, 'rb') as f:
            data = pickle.load(f)
    finally:
        gc.enable()
    return data

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def log_phase(name: str):
    print(f"\n[{name.upper()}] " + "—" * (60 - len(name)))

def serialise_subsapce(sub_config):
    identity_keys = ['layers', 'hidden_dim', 'n', 'p', 's', 'bond_dim', 'max_order']
    tags = ['method']
    if 'method' in sub_config:
        if sub_config['method'] == 'amp':
            identity_keys = ['hidden_dim', 'n', 'p', 's', 'bond_dim', 'max_order']
        parts = [f"{sub_config['method']}"]
    elif 'curry' in sub_config and sub_config['curry']:
        parts = ['cur']
    else:
        parts = []
    for key in identity_keys:
        if key in sub_config:
            val = sub_config[key]
            if type(val) is int and not isinstance(val, bool):
                label = key[:3] if len(key) > 3 else key
                parts.append(f"{label}{val}")
    return "_".join(parts)

def gen_id(config):
    q = str(config['embedding_qubits'])
    if 'discard' in config and config['discard']:
        q += 'd'
    else:
        q += 'p'
    t_str = f"t_{serialise_subsapce(config['text'])}"
    v_str = f"v_{serialise_subsapce(config['vision'])}"
    timestamp = datetime.now().strftime('%m%d_%H%M')
    return f"q{q}__{t_str}__{v_str}__{timestamp}"

def setup_exp(config_path: str):
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = get_device()
    seed = config.get("seed", int.from_bytes(os.urandom(4), "big"))
    set_seed(seed)
    
    return config, device, seed

class CheckpointManager:
    @staticmethod
    def upgrade_checkpoint(old_state_dict):
        new_state_dict = {}
        for key, value in old_state_dict.items():
            if key == "params":
                for i in range(value.size(0)):
                    new_state_dict[f"params.{i}"] = value[i:i+1]
            else:
                new_state_dict[key] = value
        return new_state_dict

    @classmethod
    def load_model(cls, checkpoint_path: Path, image_model, text_model, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        text_params = checkpoint["text"]
        image_params = checkpoint["image"]

        if "params" in text_params and not any("params." in k for k in text_params):
            text_params = cls.upgrade_checkpoint(text_params)
        if "params" in image_params and not any("params." in k for k in image_params):
            image_params = cls.upgrade_checkpoint(image_params)

        text_model.load_state_dict(text_params, strict=False)
        image_model.load_state_dict(image_params)
        
        return checkpoint