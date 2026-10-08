import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# fp16 halves VRAM for the transformer models; CPU inference stays fp32.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# MobileSAM (TinyViT encoder, ~40MB). A local copy wins; otherwise it is fetched from the Hub once.
MOBILE_SAM_LOCAL_PATH = "app/models/mobile_sam.pt"
MOBILE_SAM_HF_REPO = "dhkim2810/MobileSAM"
MOBILE_SAM_HF_FILE = "mobile_sam.pt"

# Transformers-format port of depth-anything/Depth-Anything-V2-Small.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Room photos are downscaled to this long side before any processing.
MAX_IMAGE_SIDE = 1600
