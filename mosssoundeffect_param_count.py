import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "OpenMOSS-Team/MOSS-SoundEffect"
device = "cpu"

def count_parameters(model):
    if model is None: return 0
    return sum(p.numel() for p in model.parameters())

print(f"Loading processor and model {MODEL_ID}...")

# Load the processor (which contains the audio tokenizer)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load the model
model = AutoModel.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.float32
).to(device)

print(f"\nCounting parameters for {MODEL_ID}...")

# Main MossTTSDelayModel
model_params = count_parameters(model)
print(f"Main Model (MossTTSDelayModel): {model_params:,}")

# Breakdown of the main model
if hasattr(model, 'language_model'):
    lang_params = count_parameters(model.language_model)
    print(f"  Language Model ({type(model.language_model).__name__}): {lang_params:,}")
if hasattr(model, 'emb_ext'):
    emb_params = count_parameters(model.emb_ext)
    print(f"  Extension Embeddings: {emb_params:,}")
if hasattr(model, 'lm_heads'):
    heads_params = count_parameters(model.lm_heads)
    print(f"  LM Heads: {heads_params:,}")

# Audio Tokenizer (inside the processor)
tokenizer_params = 0
if hasattr(processor, 'audio_tokenizer'):
    tokenizer_params = count_parameters(processor.audio_tokenizer)
    print(f"Audio Tokenizer ({type(processor.audio_tokenizer).__name__}): {tokenizer_params:,}")
    
    # Breakdown of the audio tokenizer
    if hasattr(processor.audio_tokenizer, 'encoder'):
         enc_params = count_parameters(processor.audio_tokenizer.encoder)
         print(f"  Encoder: {enc_params:,}")
    if hasattr(processor.audio_tokenizer, 'decoder'):
         dec_params = count_parameters(processor.audio_tokenizer.decoder)
         print(f"  Decoder: {dec_params:,}")
    if hasattr(processor.audio_tokenizer, 'quantizer'):
         quant_params = count_parameters(processor.audio_tokenizer.quantizer)
         print(f"  Quantizer: {quant_params:,}")

# Total
total_params = model_params + tokenizer_params
print(f"\nTotal parameters in {MODEL_ID}: {total_params:,}")
