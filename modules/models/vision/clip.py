import torch.nn as nn
from modules.utils.quantum_ops import amplitude_encoding

class FrozenCLIP(nn.Module):
    def __init__(self, clip_model = None, classical=True):
        super().__init__()
        self.clip_model = clip_model
        self.params = nn.ParameterList([])
        self.classical = classical

    def forward(self, img_vecs):
        if self.classical:
            return img_vecs
        else:
            return amplitude_encoding(img_vecs)