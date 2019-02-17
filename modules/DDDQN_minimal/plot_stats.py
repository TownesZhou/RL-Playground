from save_and_load import load_checkpoint
from utils import plot_durations
import matplotlib.pyplot as plt
import torch

# IMPORTANT: Set value for i_episode to indicate which checkpoint you want to use
#   for evaluation.
i_episode = 7300
start_idx = 6500
end_idx = 7200
ckpt_dir = "DDDQN_SGD_CartPoleV1_obs_checkpoints/"

input_size = 4
output_size = 2

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read checkpoint
_, _, _, training_info = \
    load_checkpoint(ckpt_dir, i_episode, input_size, output_size, device=device)

# Plot figure
plot_durations(training_info["episode reward"],
               training_info["training loss"],
               training_info["episode loss"],
               (start_idx, end_idx))