import torch
from torch.utils.data import DataLoader
from modules.utils.general import load_pkl
from modules.utils.analysis import tn_metadata, analyse_einsum
from torchvision.transforms import v2
from modules.utils.factory import build_dataset, load_obj

class DataEngine:
    def __init__(self, config, ansatz, device):
        self.config = config
        self.ansatz = ansatz
        self.device = device
        
        self.DatasetClass, self.collate_fn, self.eval_mapper = build_dataset(config)
        if config['vision']['method'] == 'amp':
            self.img_transform = v2.Compose([
                v2.ToImage(),               
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((64, 64))
            ])
        else:
            self.img_transform = None

    def compile_text(self, split):
        compile_kwargs = {}
        if self.config["model_type"] == "vqc":
            compile_kwargs["curry"] = self.config["text"].get("curry", False)
            compile_kwargs["spider"] = self.config["text"].get("spider", False)
            
        if split == 'train' or split == 'val':
            data = load_pkl(self.config["dataset"][split]["text_path"])
            return self.ansatz.compile_dataset(data, **compile_kwargs)
        elif split not in ['train', 'val']:
            data = load_pkl(self.config["dataset"]["test"][split]["text_path"])
            compiled_data = self.ansatz.compile_dataset(data, **compile_kwargs)
            return compiled_data
        else:
            raise ValueError(f"Unsupported split '{split}' for text compilation. Must be 'train' or 'val'.")

    def text_init(self, text_model, compiled_data):
        if hasattr(text_model, "from_symbols"):
            sym_cols = [c for c in compiled_data.columns if c.endswith("_symbols")]
            symbol_arr = []
            for col in sym_cols:
                symbol_arr += compiled_data[col].tolist()
            
            sym_kwargs = {"id_init": True} if self.config["model_type"] == "vqc" else {}
            text_model.from_symbols(symbol_arr, **sym_kwargs)

        if hasattr(text_model, "from_plans"):
            cols = [col for col in compiled_data.columns if col.endswith('_einsum')]
            plan_stream = []
            for col in cols:
                plan_stream += compiled_data[col].tolist()
            text_model.from_plans(list(plan_stream))
        
    def image_init(self, image_model):
        if self.config['vision']['method'] == 'pca':
            train_embeddings = torch.load(self.config["dataset"]["train"]["img_path"])
            val_embeddings = torch.load(self.config["dataset"]["val"]["img_path"])
            raw_tensor_stack = torch.stack(list(train_embeddings.values()) + list(val_embeddings.values())).to(self.device)
            image_model.fit_image_pca(raw_tensor_stack)

    def describe_einsum(self, model, data=None, return_metrics=False):
        if hasattr(model, "from_symbols"):
            sym_cols = [c for c in data.columns if c.endswith("_symbols")]
            einsum_cols = [c for c in data.columns if c.endswith("_einsum")]
            symbol_arr, einsum_arr = [], []
            for einsum_col, sym_col in zip(einsum_cols, sym_cols):
                symbol_arr += data[sym_col].tolist()
                einsum_arr += data[einsum_col].tolist()
            tn_arr = list(zip(einsum_arr, symbol_arr))
            metrics = tn_metadata(tn_arr)
            print(f"Circuit Metrics for {model.__class__.__name__}:")
            print(f"(Max) Qubits: {metrics['max'][0]:.4f} | Gates: {metrics['max'][1]:.4f} | Depth: {metrics['max'][2]:.4f} | Rank: {metrics['max'][3]:.4f} | 2-Qubit Gates: {metrics['max'][4]:.4f}")
            print(f"(Avg) Qubits: {metrics['avg'][0]:.4f} | Gates: {metrics['avg'][1]:.4f} | Depth: {metrics['avg'][2]:.4f} | Rank: {metrics['avg'][3]:.4f} | 2-Qubit Gates: {metrics['avg'][4]:.4f}")
        elif hasattr(model, "einsum_expr"):
            metrics = analyse_einsum(model.einsum_expr.replace('b', ''), model.gate_arr)
            print(f"Circuit Metrics for {model.__class__.__name__}:")
            print(f"Qubits: {metrics[0]:.4f} | Gates: {metrics[1]:.4f} | Depth: {metrics[2]:.4f} | Rank: {metrics[3]:.4f} | 2-Qubit Gates: {metrics[4]:.4f}")
        else:
            raise ValueError(f"Model {model.__class__.__name__} does not support einsum description.")
        if return_metrics:
            return metrics

    def get_dataset(self, compiled_data, split='train'):
        if split == 'train' or split == 'val':
            return self.DatasetClass(
                compiled_data, 
                self.config['dataset'][split]['img_path'], 
                image_transform=self.img_transform, 
                mode=split
            )
        elif split not in ['train', 'val']:
            BenchDatasetClass = load_obj(self.config['dataset']['test'][split].get('class', None))
            if BenchDatasetClass is None:
                raise ValueError(f"No dataset class specified for test split '{split}' in config.")
            return BenchDatasetClass(
                compiled_data, 
                self.config['dataset']['test'][split]['img_path'], 
                image_transform=self.img_transform, 
                mode='test'
            )
        else:
            raise ValueError(f"Unsupported split '{split}' for loader. Must be 'train', 'val', or a test split name.")
        
    def get_loader(self, compiled_data, split='train'):
        if split == 'train' or split == 'val':
            dataset = self.get_dataset(compiled_data, split=split)
            loader = DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                collate_fn=self.collate_fn, 
                shuffle=(split=='train'), 
                num_workers=4, 
                pin_memory=True
            )
            return loader
        elif split not in ['train', 'val']:
            dataset = self.get_dataset(compiled_data, split=split)
            collate_fn = load_obj(self.config['dataset']['test'][split]['collate_fn'])
            loader = DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                collate_fn=collate_fn, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )
            return loader
        else:
            raise ValueError(f"Unsupported split '{split}' for loader. Must be 'train', 'val', or a test split name.")