import torch
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.eval_utils import all_model_cfg

# MMAudio Models: 'small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'
model_name = 'large_44k_v2'
model_config = all_model_cfg[model_name]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

# Build the model
net = get_my_mmaudio(model_name).to(device, dtype).eval()
net.load_weights(torch.load(model_config.model_path, map_location=device, weights_only=True))

# Count the parameters
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters in {model_name}: {total_params}")
