import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OD_PT = "api/files/model/pt/ADYL11m.pt"
MODEL_IS_ONNX = "api/files/model/onnx/AD_YL11s_Seg_1.80.onnx"

MODEL_BLOCK_PT = "api/files/model/pt/AD_Productblock_YL11.pt"