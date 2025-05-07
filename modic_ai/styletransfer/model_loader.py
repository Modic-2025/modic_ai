import os
from diffusers import UNet2DConditionModel, ControlNetModel
from huggingface_hub import snapshot_download

from .StyleShot.ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline
from .StyleShot.annotator.hed import SOFT_HEDdetector
from .StyleShot.annotator.lineart import LineartDetector

device = "cuda"
models = {}
loaded_flags = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLESHOT_BASE = os.path.join(BASE_DIR, "StyleShot")
GAOJUNYAO_BASE = os.path.join(STYLESHOT_BASE, "Gaojunyao")

def ensure_download(path: str, hf_repo: str):
    if not os.path.isdir(path):
        print(f"🔽 Downloading {hf_repo} to {path}...")
        snapshot_download(hf_repo, local_dir=path)
        print(f"✅ Downloaded {hf_repo}")
    else:
        print(f"✅ Using cached: {path}")

def load_model(preprocessor: str):
    print(f"🔧 Loading model for: {preprocessor}")

    subdir = "StyleShot" if preprocessor == "Contour" else "StyleShot_lineart"
    hf_repo = "Gaojunyao/StyleShot" if preprocessor == "Contour" else "Gaojunyao/StyleShot_lineart"
    model_root = os.path.join(GAOJUNYAO_BASE, subdir)

    ensure_download(STYLESHOT_BASE, "stable-diffusion-v1-5/stable-diffusion-v1-5")
    ensure_download(STYLESHOT_BASE, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ensure_download(model_root, hf_repo)

    ip_ckpt = os.path.join(model_root, "pretrained_weight", "ip.bin")
    style_aware_encoder_path = os.path.join(model_root, "pretrained_weight", "style_aware_encoder.bin")

    base_model_path = os.path.join(STYLESHOT_BASE, "stable-diffusion-v1-5/stable-diffusion-v1-5")
    transformer_block_path = os.path.join(STYLESHOT_BASE, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet)

    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    detector = LineartDetector() if preprocessor == "Lineart" else SOFT_HEDdetector()

    models[preprocessor] = {
        "styleshot": styleshot,
        "detector": detector
    }

    loaded_flags[preprocessor] = True
    print(f"✅ Model for {preprocessor} loaded.")

def get_model(preprocessor: str):
    if not loaded_flags.get(preprocessor):
        load_model(preprocessor)
    return models[preprocessor]
