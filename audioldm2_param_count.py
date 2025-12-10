from audioldm2 import build_model
import torch

model_name = "audioldm_48k"
device = "cuda"

# Build the model
audioldm2 = build_model(model_name=model_name, device=device)

# Count the parameters
total_params = sum(p.numel() for p in audioldm2.parameters())
print(f"Total parameters in {model_name}: {total_params}")
