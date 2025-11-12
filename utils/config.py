import torch

MODEL_PATH = "src/api/files/AD_YL11s_1.0.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"