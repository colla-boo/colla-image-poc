from __future__ import annotations
from pathlib import Path
import math
import cv2
import imagehash
import numpy as np
from PIL import Image
from .ingest import open_rgb

def phash(image: Image.Image) -> str:
    return str(imagehash.phash(image))

def phash_distance(left: str, right: str) -> int:
    return imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)

def quality(image: Image.Image, weights: dict) -> dict[str, float]:
    original_short_edge = min(image.size)
    image.thumbnail((1024, 1024))
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.clip(math.log1p(lap) / math.log1p(500), 0, 1))
    mean = float(gray.mean()) / 255
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    exposure = float(np.clip((1 - abs(mean - .5) * 1.7) * (1 - clipped), 0, 1))
    contrast = float(np.clip(float(gray.std()) / 64, 0, 1))
    resolution = float(np.clip(original_short_edge / 1080, 0, 1))
    parts = dict(sharpness_score=sharpness, exposure_score=exposure,
                 contrast_score=contrast, resolution_score=resolution)
    parts['quality_score'] = sum(parts[f'{key}_score'] * weights[f'{key}_weight'] for key in ('sharpness', 'exposure', 'contrast', 'resolution'))
    return parts

def thumbnail(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((512, 512))
    image.save(target, 'WEBP', quality=82)

def device_name() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

class SiglipEncoder:
    def __init__(self, model_id: str, revision: str, labels: dict[str, str]):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device_name()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.labels = labels
        with torch.inference_mode():
            inputs = self.processor(text=list(labels.values()), padding='max_length', return_tensors='pt').to(self.device)
            vectors = self.model.get_text_features(**inputs)
            self.text_vectors = torch.nn.functional.normalize(vectors, dim=-1)

    def encode(self, paths: list[Path]) -> tuple[np.ndarray, list[tuple[str, float]]]:
        images = [open_rgb(p) for p in paths]
        with self.torch.inference_mode():
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            features = self.model.get_image_features(**inputs)
            features = self.torch.nn.functional.normalize(features, dim=-1)
            logits = features @ self.text_vectors.T
            probs = self.torch.softmax(logits * self.model.logit_scale.exp(), dim=-1).float().cpu().numpy()
            vectors = features.float().cpu().numpy()
        names = list(self.labels)
        categories = [(names[int(np.argmax(p))], float(np.max(p))) for p in probs]
        return vectors, categories
