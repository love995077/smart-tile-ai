import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_MODEL_PATH = "app/models/sam_vit_b_01ec64.pth"