import torch
import torchaudio
import soundfile as sf
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.eval_utils import generate, all_model_cfg
from mmaudio.model.utils.features_utils import FeaturesUtils
import os

ROOT_DIR="samples/mmaudio"
NUM_SAMPLES_PER_PROMPT = 1
RND_BASE = 12345

# Generator Parameter Values
GEN_PARAM_NUM_INFERENCE_STEPS = 100
GEN_PARAM_LEN_IN_SEC = 10.0
GEN_PARAM_NUM_WAVEFORMS_PER_PROMPT = 1

# Output File Value
GEN_AUDIO_SAMPLE_RATE = 44100
GEN_AUDIO_DURATION_IN_S = 10

def prompt_to_filename(prompt):
  underscore_name = prompt.replace(" ", "_").lower()
  return underscore_name

# MMAudio Models: 'small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'

if __name__ == "__main__":
    ## Model Setup
    with torch.no_grad():
      model_name = 'large_44k_v2'
      model_config = all_model_cfg[model_name]
      model_config.download_if_needed()

      seq_cfg = model_config.seq_cfg
      seq_cfg.duration = GEN_AUDIO_DURATION_IN_S
      seq_cfg.sampling_rate = GEN_AUDIO_SAMPLE_RATE

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      dtype = torch.bfloat16

      net = get_my_mmaudio(model_name).to(device, dtype).eval()
      net.load_weights(torch.load(model_config.model_path, map_location=device, weights_only=True))

      feature_utils = FeaturesUtils(
          tod_vae_ckpt=model_config.vae_path,
          synchformer_ckpt=model_config.synchformer_ckpt,
          enable_conditions=True,
          mode=model_config.mode,
          bigvgan_vocoder_ckpt=model_config.bigvgan_16k_path,
          need_vae_encoder=False
      ).to(device, dtype).eval()

      net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

      ## Main Logic for single prompt
      prompt = "Dog barking" # Single prompt for generation
      gen_audio_name = prompt_to_filename(prompt)
      audio_class_folder = f"{ROOT_DIR}/{gen_audio_name}"
      if not os.path.exists(audio_class_folder):
        os.makedirs(audio_class_folder)

      NUM_SAMPLES_PER_PROMPT = 3 # Update to 3 samples
      
      for i in range(NUM_SAMPLES_PER_PROMPT):
          random_seed = RND_BASE + i
          print(f"Generating sample {i+1}/{NUM_SAMPLES_PER_PROMPT} with seed {random_seed}...")
          
          rng = torch.Generator(device=device)
          rng.manual_seed(random_seed)
          fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=100)

          cfg_strength = 4.5
          clip_frames = None
          sync_frames = None

          audios = generate(
              clip_frames,
              sync_frames,
              [prompt],
              feature_utils=feature_utils,
              net=net,
              fm=fm,
              rng=rng,
              cfg_strength=cfg_strength
          )

          audio = audios.float().cpu()[0]
          generated_output = audio.T.numpy()
          gen_audio_filename_path = f"{audio_class_folder}/{i+1:03d}.wav" 
          sf.write(gen_audio_filename_path, generated_output, seq_cfg.sampling_rate)