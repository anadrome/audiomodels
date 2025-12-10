from huggingface_hub import login
import torch
import os
import soundfile as sf

from diffusers import StableAudioPipeline

hf_key = os.environ.get("HF_TOKEN")
if not hf_key:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
login(hf_key)

ROOT_DIR="samples/stableaudio"
NUM_SAMPLES_PER_PROMPT = 1
RND_BASE = 12345

# Generator Parameter Values
GEN_PARAM_NUM_INFERENCE_STEPS = 100
GEN_PARAM_LEN_IN_SEC = 10.0
GEN_PARAM_NUM_WAVEFORMS_PER_PROMPT = 1

# Output File Value
GEN_AUDIO_SAMPLE_RATE = 44100

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def prompt_to_folderfile_name(prompt):
  underscore_name = prompt.replace(" ", "_").lower()
  return underscore_name

if __name__ == "__main__":
    prompt = "Dog barking" # Single prompt for generation
    label = prompt_to_folderfile_name(prompt)
    gen_audio_folder = f"{ROOT_DIR}/{label}"

    if not os.path.exists(gen_audio_folder):
        os.makedirs(gen_audio_folder)

    NUM_SAMPLES_PER_PROMPT = 3 # Update to 3 samples

    for i in range(NUM_SAMPLES_PER_PROMPT):
        random_seed = RND_BASE + i
        print(f"Generating sample {i+1}/{NUM_SAMPLES_PER_PROMPT} with seed {random_seed}...")
        
        generator = torch.Generator("cuda").manual_seed(random_seed)        # Generator Random Seed

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
