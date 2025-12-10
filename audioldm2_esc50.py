sounds = [
    "Dog", "Rain", "Crying baby", "Door knock", "Helicopter",
    "Rooster", "Sea waves", "Sneezing", "Mouse click", "Chainsaw",
    "Pig", "Crackling fire", "Clapping", "Keyboard typing", "Siren",
    "Cow", "Crickets", "Breathing", "Door, wood creaks", "Car horn",
    "Frog", "Chirping birds", "Coughing", "Can opening", "Engine",
    "Cat", "Water drops", "Footsteps", "Washing machine", "Train",
    "Hen", "Wind", "Laughing", "Vacuum cleaner", "Church bells",
    "Insects (flying)", "Pouring water", "Brushing teeth", "Clock alarm", "Airplane",
    "Sheep", "Toilet flush", "Snoring", "Clock tick", "Fireworks",
    "Crow", "Thunderstorm", "Drinking, sipping", "Glass breaking", "Hand saw"
]
import subprocess

ROOT_DIR="samples/audioldm2"
NUM_SAMPLES_PER_PROMPT = 100
RND_BASE = 12340
MODEL="audioldm_48k"

mode = "cuda"
for item in sounds:
  for c in range(100):
    seed = RND_BASE + c
    x = f'audioldm2 -t "Sound of {item}" --model {MODEL} --seed {seed} --ddim_steps 200 -d {mode} -s {ROOT_DIR}'
    subprocess.run(x, shell=True)
