import torch
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.eval_utils import all_model_cfg
from mmaudio.model.utils.features_utils import FeaturesUtils

# MMAudio Models: 'small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'
model_name = 'large_44k_v2'
model_config = all_model_cfg[model_name]

# Ensure weights are available
model_config.download_if_needed()

device = 'cpu'
dtype = torch.float32

print(f"Counting parameters for {model_name}...")

# Main diffusion transformer (DiT)
net = get_my_mmaudio(model_name).to(device, dtype).eval()
net.load_weights(torch.load(model_config.model_path, map_location=device, weights_only=True))
dit_params = sum(p.numel() for p in net.parameters())
print(f"Diffusion transformer (DiT): {dit_params:,}")

# Additional networks from FeaturesUtils: CLIP, Synchformer, VAE, Vocoder
# Load with need_vae_encoder=False because that's what we use for inference.
feature_utils = FeaturesUtils(
    tod_vae_ckpt=model_config.vae_path,
    synchformer_ckpt=model_config.synchformer_ckpt,
    enable_conditions=True,
    mode=model_config.mode,
    bigvgan_vocoder_ckpt=model_config.bigvgan_16k_path,
    need_vae_encoder=False 
).to(device, dtype).eval()

# CLIP
clip_params = 0
if feature_utils.clip_model is not None:
    clip_params = sum(p.numel() for p in feature_utils.clip_model.parameters())
    print(f"CLIP: {clip_params:,}")

# Synchformer
sync_params = 0
if feature_utils.synchformer is not None:
    sync_params = sum(p.numel() for p in feature_utils.synchformer.parameters())
    print(f"Synchformer: {sync_params:,}")

# VAE
vae_params = 0
if feature_utils.tod is not None and feature_utils.tod.vae is not None:
    vae_params = sum(p.numel() for p in feature_utils.tod.vae.parameters())
    print(f"VAE (Decoder only): {vae_params:,}")

# Vocoder
vocoder_params = 0
if feature_utils.tod is not None and feature_utils.tod.vocoder is not None:
    vocoder_params = sum(p.numel() for p in feature_utils.tod.vocoder.parameters())
    print(f"Vocoder (BigVGAN): {vocoder_params:,}")

total_params = dit_params + clip_params + sync_params + vae_params + vocoder_params
print(f"Total parameters in {model_name}: {total_params:,}")
