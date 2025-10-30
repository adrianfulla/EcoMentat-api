import io
import os
import base64
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from typing import Tuple

# If your project expects this name/order, lock it here:
ATMOS_KEYS = ["NO2", "CO", "PM2.5", "PM10"]

_eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_ATMOS_MEAN = None
_ATMOS_STD = None
_THRESH = 0.5

def _load_zscore(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    return z["mean"], z["std"]

def _z_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.clip(std, 1e-6, None)

def _image_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return img

def _image_from_base64(s: str) -> Image.Image:
    return _image_from_bytes(base64.b64decode(s))

def _ensure_model_loaded():
    global _MODEL, _ATMOS_MEAN, _ATMOS_STD, _THRESH

    if _MODEL is not None:
        return

    # Read env
    visual_ckpt = os.environ["VISUAL_CKPT_PATH"]
    final_weights = os.environ["FINAL_WEIGHTS"]
    atmos_z_path = os.environ["ATMOS_Z_PATH"]
    microbatch = int(os.environ.get("VISUAL_MICROBATCH", "2"))
    _THRESH = float(os.environ.get("THRESH", "0.5"))

    # Lazy-import here to avoid import costs if unused
    from architecture.multimodal_arch import build_model

    # Build + load weights
    _MODEL = build_model(visual_ckpt, visual_microbatch=microbatch).to(_DEVICE)
    sd = torch.load(final_weights, map_location="cpu", weights_only=True)
    _MODEL.load_state_dict(sd, strict=True)
    _MODEL.eval()

    # Atmos normalization
    mean, std = _load_zscore(atmos_z_path)
    _ATMOS_MEAN = mean.astype(np.float32)
    _ATMOS_STD = std.astype(np.float32)

@torch.no_grad()
def predict_from_image_and_atmos(img: Image.Image, atmos_ordered: list[float]) -> Tuple[float, int]:
    """
    img: PIL.Image (RGB)
    atmos_ordered: [NO2, CO, PM2.5, PM10]
    """
    _ensure_model_loaded()

    x_img = _eval_tf(img).unsqueeze(0).to(_DEVICE)       # [1,3,224,224]
    x_at = np.asarray(atmos_ordered, dtype=np.float32)   # [4]
    x_at_z = _z_transform(x_at, _ATMOS_MEAN, _ATMOS_STD)
    x_at_t = torch.from_numpy(x_at_z).unsqueeze(0).to(_DEVICE)  # [1,4]

    use_amp = (_DEVICE == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = _MODEL(x_img, x_at_t)  # [1] or [1,1]
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim > 1:
            logits = logits.view(-1)
        prob = torch.sigmoid(logits)[0].item()

    label = int(prob >= _THRESH)
    return float(prob), label

def predict_from_base64(image_b64: str, atmos_ordered: list[float]) -> Tuple[float, int]:
    img = _image_from_base64(image_b64)
    return predict_from_image_and_atmos(img, atmos_ordered)

def predict_from_bytes(image_bytes: bytes, atmos_ordered: list[float]) -> Tuple[float, int]:
    img = _image_from_bytes(image_bytes)
    return predict_from_image_and_atmos(img, atmos_ordered)
