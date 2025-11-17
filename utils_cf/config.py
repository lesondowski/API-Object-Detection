import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OD_PT = "api/files/AD_YL11s_1.0.pt"
MODEL_IS_ONNX = "api/files/AD_YL11s_Seg_1.80.onnx"
