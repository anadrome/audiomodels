import subprocess
import os
import shutil
import glob

# User input here (prompt & number of samples)
prompt = "dramatic sound of glass breaking, cinematic, high-fidelity"
NUM_SAMPLES_PER_PROMPT = 3

ROOT_DIR="samples/audioldm2"
RND_BASE = 12340
MODEL="audioldm_48k"

def prompt_to_folder_name(prompt):
    return prompt.replace(" ", "_").lower()

target_dir = os.path.join(ROOT_DIR, prompt_to_folder_name(prompt))
os.makedirs(target_dir, exist_ok=True)

import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

for i in range(NUM_SAMPLES_PER_PROMPT):
    seed = RND_BASE + i
    print(f"Generating sample {i+1}/{NUM_SAMPLES_PER_PROMPT} with seed {seed}...")
    
    # Generate via separate CLI script
    x = f'audioldm2 -t "{prompt}" --model {MODEL} --seed {seed} --ddim_steps 200 -d {device} -s {ROOT_DIR}'
    subprocess.run(x, shell=True)
    
    # Post-processing:
    # AudioLDM2 creates a new folder named with a timestamp for each run.
    # Look for the most recently created folder, move the file out, and delete it.
    subdirs = glob.glob(os.path.join(ROOT_DIR, "*"))
    candidate_dirs = [d for d in subdirs if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(target_dir)]
    
    if candidate_dirs:
        latest_dir = max(candidate_dirs, key=os.path.getctime)
        source_file = os.path.join(latest_dir, f"{prompt}.wav")
        
        if os.path.exists(source_file):
            dest_file = os.path.join(target_dir, f"{i+1:03d}.wav")
            print(f"Moving {source_file} to {dest_file}")
            shutil.move(source_file, dest_file)
            os.rmdir(latest_dir)
        else:
            print(f"Warning: Expected generated file {source_file} not found.")
    else:
        print("Warning: No output directory found from AudioLDM2.")
