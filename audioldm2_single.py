import subprocess
import os
import shutil
import glob

ROOT_DIR="samples/audioldm2"
NUM_SAMPLES_PER_PROMPT = 3
RND_BASE = 12340
MODEL="audioldm_48k"

prompt = "Dog barking" # Single prompt for generation

def prompt_to_folder_name(prompt):
    return prompt.replace(" ", "_").lower()

target_dir = os.path.join(ROOT_DIR, prompt_to_folder_name(prompt))
os.makedirs(target_dir, exist_ok=True)

mode = "cuda"

for i in range(NUM_SAMPLES_PER_PROMPT):
    seed = RND_BASE + i
    print(f"Generating sample {i+1}/{NUM_SAMPLES_PER_PROMPT} with seed {seed}...")
    
    # Run generation
    x = f'venv_audioldm2/bin/audioldm2 -t "{prompt}" --model {MODEL} --seed {seed} --ddim_steps 200 -d {mode} -s {ROOT_DIR}'
    subprocess.run(x, shell=True)
    
    # Post-processing: Find and move the generated file
    # AudioLDM2 creates a new folder for each run inside ROOT_DIR
    # We look for the most recently created folder
    subdirs = glob.glob(os.path.join(ROOT_DIR, "*"))
    # Filter only directories that look like timestamps (optional, but safer if there are other files)
    # For simplicity, just finding the newest directory that isn't our target_dir
    candidate_dirs = [d for d in subdirs if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(target_dir)]
    
    if candidate_dirs:
        latest_dir = max(candidate_dirs, key=os.path.getctime)
        source_file = os.path.join(latest_dir, f"{prompt}.wav")
        
        if os.path.exists(source_file):
            dest_file = os.path.join(target_dir, f"{i+1:03d}.wav")
            print(f"Moving {source_file} to {dest_file}")
            shutil.move(source_file, dest_file)
            
            # Clean up the empty directory
            try:
                os.rmdir(latest_dir)
            except OSError:
                print(f"Warning: Could not remove directory {latest_dir}")
        else:
            print(f"Warning: Expected generated file {source_file} not found.")
    else:
        print("Warning: No output directory found from AudioLDM2.")
