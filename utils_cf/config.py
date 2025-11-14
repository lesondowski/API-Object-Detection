import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PIS= "api/files/AD_YL8n_Seg_1.0.pt" 
MODEL_POD = "api/files/AD_YL11s_1.0.pt"