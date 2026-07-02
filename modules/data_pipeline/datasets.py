from __future__ import annotations
import zipfile, io, torch, random
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset

def subsample_data(dataset, nsamp=0):
    # Using this function without specifying nsamp is equivalent to a random shuffle of the data
    if nsamp == 0: 
        nsamp = len(dataset)
    idx_arr = random.sample(range(0, len(dataset)), nsamp)
    return Subset(dataset, idx_arr)

class BaseDataset(Dataset):
    def __init__(self, compiled_df: pd.DataFrame, img_path: str | None, image_transform=None, mode: str = "train"):
        self.compiled_df = compiled_df
        self.image_transform = image_transform
        self.img_path = img_path
        self.img_store = img_path.split('.')[-1].lower() if img_path is not None else 'df'
        self.mode = mode.lower()

        if self.img_store == 'zip':
            with zipfile.ZipFile(self.img_path, 'r') as archive:
                # self.archive = zipfile.ZipFile(img_path, 'r')
                self.namelist_set = set(archive.namelist())
                self.parent = archive.namelist()[0]
                self.suffix = Path(archive.namelist()[1]).suffix
            self.archive = None
        elif self.img_store == 'pt':
            self.embeddings = torch.load(img_path)

    def __len__(self):
        return len(self.compiled_df) 

    def _load_image(self, image_input: str) -> Image.Image:
        if pd.isna(image_input) or image_input is None:
            raise ValueError("Encountered empty or missing image identifier reference row.")
        
        if self.img_store == 'pt':
            return self.embeddings[int(image_input)]
        elif self.img_store == 'df':
            img = image_input.convert("RGB")
        elif self.img_store == 'zip':
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.img_path, 'r')
            zip_path = str(image_input)
            if not zip_path.startswith(self.parent):
                zip_path = self.parent + zip_path
            if not zip_path.endswith(self.suffix):
                zip_path += self.suffix
            img_bytes = self.archive.read(zip_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            raise ValueError(f"Unsupported layout configuration '{self.img_store}'.")

        if self.image_transform:
            img = self.image_transform(img)
        return img

class CocoDataset(BaseDataset):
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load the single image reference
        image_ref = row_compiled.get('file_name')
        image = self._load_image(image_ref)
        
        # Compiled data contains lists for columns with multiple captions
        einsums = row_compiled['captions_einsum']
        symbols = row_compiled['captions_symbols']
        
        # During contrastive training, pick one caption variation randomly per sample
        if self.mode == "train":
            caption_idx = random.randint(0, len(einsums) - 1)
        
            return {"image": image,
                    "caption": (einsums[caption_idx], symbols[caption_idx])
                    }
        else:
            return {"image": image,
                    "caption": list(zip(einsums, symbols))
                    }

def coco_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch] 
    
    return {
        "image": images,
        "caption": captions
    }

class ARODataset(BaseDataset):
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load image
        image = self._load_image(row_compiled['image_id'])
        
        # Extract both expressions side-by-side for comparison scoring
        pos_einsum = row_compiled['true_caption_einsum']
        pos_symbols = row_compiled['true_caption_symbols']
        
        neg_einsum = row_compiled['false_caption_einsum']
        neg_symbols = row_compiled['false_caption_symbols']
        
        return {"image": image,
                "pos_caption": (pos_einsum, pos_symbols),
                "neg_caption": (neg_einsum, neg_symbols)
                }
    
def aro_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    pos_captions = [item['pos_caption'] for item in batch] 
    neg_captions = [item['neg_caption'] for item in batch]
    
    return {
        "image": images,
        "pos_caption": pos_captions,
        "neg_caption": neg_captions
    }
    
class SVODataset(BaseDataset):
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load image
        pos_image = self._load_image(row_compiled['pos_image_id'])
        neg_image = self._load_image(row_compiled['neg_image_id'])
        
        # Extract both expressions side-by-side for comparison scoring
        einsum = row_compiled['corrected_sentence_einsum']
        symbols = row_compiled['corrected_sentence_symbols']
        
        return {"caption": (einsum, symbols),
                "pos_image": pos_image,
                "neg_image": neg_image
                }

def svo_collate_fn(batch):
    captions = [item['caption'] for item in batch]
    pos_images = torch.stack([item['pos_image'] for item in batch])
    neg_images = torch.stack([item['neg_image'] for item in batch])
    
    return {
        "caption": captions,
        "pos_image": pos_images,
        "neg_image": neg_images
    }

class SwapDataset(BaseDataset):
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load image
        image = self._load_image(row_compiled['pos_image_id'])
        
        # Extract both expressions side-by-side for comparison scoring
        pos_einsum = row_compiled['corrected_sentence_einsum']
        pos_symbols = row_compiled['corrected_sentence_symbols']

        neg_einsum = row_compiled['swapped_sentence_einsum']
        neg_symbols = row_compiled['swapped_sentence_symbols']
        
        return {"image": image,
                "pos_caption": (pos_einsum, pos_symbols),
                "neg_caption": (neg_einsum, neg_symbols)
                }
    
def swap_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    pos_captions = [item['pos_caption'] for item in batch] 
    neg_captions = [item['neg_caption'] for item in batch]
    
    return {
        "image": images,
        "pos_caption": pos_captions,
        "neg_caption": neg_captions
    }

class SugarCrepePPDataset(BaseDataset):
    """Handles SugarCrepe / ARO layout: 1 Image -> (Positive Caption vs Negative Caption)."""
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load image
        image_ref = row_compiled.get('filename')
        image = self._load_image(image_ref)
        
        # Extract both expressions side-by-side for comparison scoring
        pos1 = (row_compiled['caption_einsum'], row_compiled['caption_symbols'])
        pos2 = (row_compiled['caption2_einsum'], row_compiled['caption2_symbols'])
        neg = (row_compiled['negative_caption_einsum'], row_compiled['negative_caption_symbols'])
        
        return {"image": image,
                "pos_caption1": pos1,
                "pos_caption2": pos2,
                "neg_caption": neg
                }

def sugarcrepe_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    pos1_captions = [item['pos_caption1'] for item in batch] 
    pos2_captions = [item['pos_caption2'] for item in batch]
    neg_captions = [item['neg_caption'] for item in batch]
    
    return {
        "image": images,
        "pos_caption1": pos1_captions,
        "pos_caption2": pos2_captions,
        "neg_caption": neg_captions
    }

class WinoGroundDataset(BaseDataset):
    """Handles Winoground layout: 2 Images and 2 Captions per test group."""
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Winoground contains 2 separate image columns per row
        img_0 = self._load_image(row_compiled['image_0'])
        img_1 = self._load_image(row_compiled['image_1'])
        
        # Extract both distinct compiled caption blueprints
        c0_einsum = row_compiled['caption_0_einsum']
        c0_symbols = row_compiled['caption_0_symbols']
        
        c1_einsum = row_compiled['caption_1_einsum']
        c1_symbols = row_compiled['caption_1_symbols']
        
        return {'image_0': img_0,
                'image_1': img_1,
                'caption_0': (c0_einsum, c0_symbols),
                'caption_1': (c1_einsum, c1_symbols)
            }
    
def winoground_collate_fn(batch):
    img_0 = torch.stack([item['image_0'] for item in batch])
    img_1 = torch.stack([item['image_1'] for item in batch])
    caption_0 = [item['caption_0'] for item in batch] 
    caption_1 = [item['caption_1'] for item in batch]
    
    return {
        'image_0': img_0,
        'image_1': img_1,
        'caption_0': caption_0,
        'caption_1': caption_1
    }