import torch
import os
import soundfile as sf

from diffusers import StableAudioPipeline

ROOT_DIR="samples/stableaudio"
NUM_SAMPLES_PER_PROMPT = 100
RND_BASE = 12345

# Generator Parameter Values
GEN_PARAM_NUM_INFERENCE_STEPS = 100
GEN_PARAM_LEN_IN_SEC = 10.0
GEN_PARAM_NUM_WAVEFORMS_PER_PROMPT = 1

# Output File Value
GEN_AUDIO_SAMPLE_RATE = 44100

esc_50_labels = {
    "Animals": ["Dog", "Rooster", "Pig", "Cow", "Frog", "Cat", "Hen", "Insects (flying)", "Sheep", "Crow"],
    "Natural soundscapes & water sounds": ["Rain", "Sea waves", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind", "Pouring water", "Toilet flush", "Thunderstorm"],
    "Human, non-speech sounds": ["Crying baby", "Sneezing", "Clapping", "Breathing", "Coughing", "Footsteps", "Laughing", "Brushing teeth", "Snoring", "Drinking, sipping"],
    "Interior/domestic sounds": ["Door knock", "Mouse click", "Keyboard typing", "Door, wood creaks", "Can opening", "Washing machine", "Vacuum cleaner", "Clock alarm", "Clock tick", "Glass breaking"],
    "Exterior/urban noises": ["Helicopter", "Chainsaw", "Siren", "Car horn", "Engine", "Train", "Church bells", "Airplane", "Fireworks", "Hand saw"]
}

def get_esc_50_level_1_labels(esc_50_labels):
    level_1_labels = list(esc_50_labels.keys())
    return level_1_labels

def get_esc_50_level_2_labels(esc_50_labels):
    level_2_labels = [label for sublist in esc_50_labels.values() for label in sublist]
    return level_2_labels


esc_50_level_1_labels = get_esc_50_level_1_labels(esc_50_labels)
esc_50_level_2_labels = get_esc_50_level_2_labels(esc_50_labels)

import sys
from diffusers import DPMSolverMultistepScheduler

if sys.platform == "darwin":
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
else:
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=dtype)

pipe = pipe.to(device)

def prompt_to_folderfile_name(prompt):
  underscore_name = prompt.replace(" ", "_").lower()
  return underscore_name

if __name__ == "__main__":


    for prompt_index in range(len(esc_50_level_2_labels)):                                # for each prompt
        folder_name = prompt_to_folderfile_name(esc_50_level_2_labels[prompt_index])
        gen_audio_folder = f"{ROOT_DIR}/{folder_name}"

        for gen_audio_index in range(NUM_SAMPLES_PER_PROMPT):                   # for each audio file to generate per prompt
            if gen_audio_index == 0:
                if not os.path.exists(gen_audio_folder):
                    os.makedirs(gen_audio_folder)


            random_seed = RND_BASE + prompt_index + gen_audio_index
            generator = torch.Generator(device).manual_seed(random_seed)        # Generator Random Seed

            audio = pipe(
                "Sound of " + esc_50_level_2_labels[prompt_index],
                num_inference_steps=GEN_PARAM_NUM_INFERENCE_STEPS,
                audio_end_in_s= GEN_PARAM_LEN_IN_SEC,
                num_waveforms_per_prompt=GEN_PARAM_NUM_WAVEFORMS_PER_PROMPT,
                generator=generator,
            ).audios

            generated_output = audio[0].T.float().cpu().numpy()
            generated_output_filename = f"{gen_audio_index+1:03d}.wav"
            generated_output_abs_path = f"{gen_audio_folder}/{generated_output_filename}"
            print(f"generated_output_filename = {generated_output_filename}")
            print(f"generated_output_abs_path = {generated_output_abs_path}")


            sf.write(generated_output_abs_path, generated_output, GEN_AUDIO_SAMPLE_RATE)

