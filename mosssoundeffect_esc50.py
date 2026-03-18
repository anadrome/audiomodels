import torch
import soundfile as sf
import os
from transformers import AutoModel, AutoProcessor

ROOT_DIR = "samples/mosssoundeffect"
NUM_SAMPLES_PER_PROMPT = 100
RND_BASE = 12345
MODEL_ID = "OpenMOSS-Team/MOSS-SoundEffect"

esc_50_labels = {
    "Animals": ["Dog", "Rooster", "Pig", "Cow", "Frog", "Cat", "Hen", "Insects (flying)", "Sheep", "Crow"],
    "Natural soundscapes & water sounds": ["Rain", "Sea waves", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind", "Pouring water", "Toilet flush", "Thunderstorm"],
    "Human, non-speech sounds": ["Crying baby", "Sneezing", "Clapping", "Breathing", "Coughing", "Footsteps", "Laughing", "Brushing teeth", "Snoring", "Drinking, sipping"],
    "Interior/domestic sounds": ["Door knock", "Mouse click", "Keyboard typing", "Door, wood creaks", "Can opening", "Washing machine", "Vacuum cleaner", "Clock alarm", "Clock tick", "Glass breaking"],
    "Exterior/urban noises": ["Helicopter", "Chainsaw", "Siren", "Car horn", "Engine", "Train", "Church bells", "Airplane", "Fireworks", "Hand saw"]
}

def get_esc_50_level_2_labels(esc_50_labels):
    level_2_labels = [label for sublist in esc_50_labels.values() for label in sublist]
    return level_2_labels

esc_50_level_2_labels = get_esc_50_level_2_labels(esc_50_labels)

def prompt_to_filename(prompt):
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

    for i, label in enumerate(esc_50_level_2_labels):
        gen_audio_name = prompt_to_filename(label)
        audio_class_folder = os.path.join(ROOT_DIR, gen_audio_name)
        os.makedirs(audio_class_folder, exist_ok=True)
        
        prompt = "Sound of " + label
        print(f"Generating samples for: {prompt}")

        for j in range(NUM_SAMPLES_PER_PROMPT):
            random_seed = RND_BASE + i + j
            torch.manual_seed(random_seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(random_seed)
            
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
                    
                    dest_file = os.path.join(audio_class_folder, f"{j+1:03d}.wav")
                    
                    # Ensure audio_tensor is on CPU and has correct shape (T, C) for soundfile
                    audio_np = audio_tensor.float().cpu().numpy()
                    if audio_np.ndim == 2:
                        audio_np = audio_np.T # (C, T) -> (T, C)
                    
                    sf.write(dest_file, audio_np, sampling_rate)
                else:
                    print(f"Warning: No audio codes found for {label} sample {j+1}")
