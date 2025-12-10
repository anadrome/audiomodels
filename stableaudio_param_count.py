import torch
from diffusers import StableAudioPipeline

# Load the pipeline
pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Count the parameters
total_params = 0
for component in [pipe.vae, pipe.text_encoder, pipe.transformer, pipe.projection_model]:
    if component is not None:
        total_params += sum(p.numel() for p in component.parameters())

print(f"Total parameters in stable-audio-open-1.0: {total_params}")
