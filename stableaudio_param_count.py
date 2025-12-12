import torch
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

total_params = 0
components = [
    ("VAE", pipe.vae),
    ("Text encoder", pipe.text_encoder),
    ("Transformer", pipe.transformer),
    ("Projection model", pipe.projection_model)
]

for name, component in components:
    if component is not None:
        count = sum(p.numel() for p in component.parameters())
        print(f"{name}: {count:,} parameters")
        total_params += count

print(f"Total parameters in stable-audio-open-1.0: {total_params:,}")
