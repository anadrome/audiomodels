from audioldm2 import build_model
import torch

model_name = "audioldm_48k"
device = "cpu"

audioldm2 = build_model(model_name=model_name, device=device)

def count_parameters(model):
    if model is None: return 0
    return sum(p.numel() for p in model.parameters())

print(f"Counting parameters for {model_name}...")

# U-Net
unet_params = count_parameters(audioldm2.model)
print(f"U-Net (DiffusionWrapper): {unet_params:,}")

# VAE
vae_params = 0
if hasattr(audioldm2, 'first_stage_model'):
    vae_params = count_parameters(audioldm2.first_stage_model)
    print(f"VAE (first stage): {vae_params:,}")

# Conditioning models
cond_params = 0
if hasattr(audioldm2, 'cond_stage_models') and audioldm2.cond_stage_models is not None:
    for i, model in enumerate(audioldm2.cond_stage_models):
        p = count_parameters(model)
        print(f"Conditioning model {i} ({type(model).__name__}): {p:,}")
        cond_params += p

# CLAP
clap_params = 0
if hasattr(audioldm2, 'clap'):
    clap_params = count_parameters(audioldm2.clap)
    print(f"CLAP: {clap_params:,}")

# Total
total_params = count_parameters(audioldm2)
print(f"Total parameters in {model_name}: {total_params:,}")
