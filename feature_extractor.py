# src/feature_extractor.py
import torch
from torchvision import models

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # Carga VGG19 preentrenado
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(self.device)
        self.model.eval()
        # Índices después de cada MaxPool en VGG19
        self.max_pooling_indices = [9, 18, 27, 36]

    @torch.no_grad()
    def extract_feature_maps(self, x, stage=0):
        x = x.to(self.device)
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if i == self.max_pooling_indices[stage]:
                return x
        raise ValueError(f"Stage {stage} inválido")

    def extract_flattened(self, x, stage=0):
        fmap = self.extract_feature_maps(x, stage=stage)
        return torch.flatten(fmap, start_dim=1)
