import torch
import os
import soundfile as sf
import sys
from diffusers import StableAudioPipeline, DPMSolverMultistepScheduler

# User input here (prompt & number of samples)
prompt = "dramatic sound of glass breaking, cinematic, high-fidelity"
NUM_SAMPLES_PER_PROMPT = 3

ROOT_DIR="samples/stableaudio"
RND_BASE = 12345

# Generator Parameter Values
GEN_PARAM_NUM_INFERENCE_STEPS = 100
GEN_PARAM_LEN_IN_SEC = 10.0
GEN_PARAM_NUM_WAVEFORMS_PER_PROMPT = 1
GEN_AUDIO_SAMPLE_RATE = 44100

if sys.platform == "darwin":
    # macOS: Use MPS if available, and a different scheduler to avoid torchsde recursion issues
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
else:
    # Linux/Other: Use CUDA if available and default scheduler
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=dtype)

pipe = pipe.to(device)

def prompt_to_filename(prompt):
  underscore_name = prompt.replace(" ", "_").lower()
  return underscore_name

if __name__ == "__main__":
    label = prompt_to_filename(prompt)
    gen_audio_folder = f"{ROOT_DIR}/{label}"

    if not os.path.exists(gen_audio_folder):
        os.makedirs(gen_audio_folder)

    for i in range(NUM_SAMPLES_PER_PROMPT):
        random_seed = RND_BASE + i
        print(f"Generating sample {i+1}/{NUM_SAMPLES_PER_PROMPT} with seed {random_seed}...")
        
        generator = torch.Generator(device).manual_seed(random_seed)

        audio = pipe(
            prompt,
            num_inference_steps=GEN_PARAM_NUM_INFERENCE_STEPS,
            audio_end_in_s= GEN_PARAM_LEN_IN_SEC,
            num_waveforms_per_prompt=GEN_PARAM_NUM_WAVEFORMS_PER_PROMPT,
            generator=generator,
        ).audios

        generated_output = audio[0].T.float().cpu().numpy()
        generated_output_filename = f"{i+1:03d}.wav"
        generated_output_abs_path = f"{gen_audio_folder}/{generated_output_filename}"
        print(f"generated_output_filename = {generated_output_filename}")
        print(f"generated_output_abs_path = {generated_output_abs_path}")

        sf.write(generated_output_abs_path, generated_output, GEN_AUDIO_SAMPLE_RATE)
