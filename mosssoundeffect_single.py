import torch
import soundfile as sf
import os
from transformers import AutoModel, AutoProcessor

# User input here (prompt & number of samples)
prompt = "dramatic sound of glass breaking, cinematic, high-fidelity"
NUM_SAMPLES_PER_PROMPT = 3

ROOT_DIR = "samples/mosssoundeffect"
RND_BASE = 12345
MODEL_ID = "OpenMOSS-Team/MOSS-SoundEffect"

def prompt_to_folder_name(prompt):
    return prompt.replace(" ", "_").lower().replace(",", "")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {MODEL_ID} on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    target_dir = os.path.join(ROOT_DIR, prompt_to_folder_name(prompt))
    os.makedirs(target_dir, exist_ok=True)

    for i in range(NUM_SAMPLES_PER_PROMPT):
        seed = RND_BASE + i
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
            
        print(f"Generating sample {i+1}/{NUM_SAMPLES_PER_PROMPT} with seed {seed}...")
        
        # The model expects a conversation format
        conversation = [[processor.build_user_message(ambient_sound=prompt)]]
        inputs = processor(conversation, mode="generation")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=144, # 10 seconds
            )
            
        decoded_outputs = processor.decode(outputs)
        for message in decoded_outputs:
            if hasattr(message, 'audio_codes_list') and message.audio_codes_list:
                audio_tensor = message.audio_codes_list[0]
                sampling_rate = processor.model_config.sampling_rate
                
                dest_file = os.path.join(target_dir, f"{i+1:03d}.wav")
                # Ensure audio_tensor is on CPU and has correct shape (T, C) for soundfile
                audio_np = audio_tensor.float().cpu().numpy()
                if audio_np.ndim == 2:
                    audio_np = audio_np.T # (C, T) -> (T, C)
                
                sf.write(dest_file, audio_np, sampling_rate)
                print(f"Saved to {dest_file}")
            else:
                print("Warning: No audio codes found in decoded message.")
