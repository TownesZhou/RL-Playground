import os
import torch
from torch import optim
from model_DDDQN import DQN

# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, policy_net, target_net, optimizer, i_episode, learning_rate=0.001, **kwargs):
    save_dict = {"policy_net": policy_net.state_dict(),
                 "target_net": target_net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "i_episode": i_episode,
                 "learning_rate": learning_rate
                 }
    # Save optional contents
    save_dict.update(kwargs)

    # Create the directory if not exist
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    file_name = os.path.join(file_dir, "ckpt_eps%d.pt" % i_episode)

    # Delete the file if the file already exist
    try:
        os.remove(file_name)
    except OSError:
        pass

    # Save the file
    torch.save(save_dict, file_name)


def load_checkpoint(file_dir, i_episode, input_size, output_size, device='cuda'):
    checkpoint = torch.load(os.path.join(file_dir, "ckpt_eps%d.pt" % i_episode))

    policy_net = DQN(input_size, output_size).to(device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.train()

    target_net = DQN(input_size, output_size).to(device)
    target_net.load_state_dict(checkpoint["target_net"])
    target_net.eval()

    learning_rate = checkpoint["learning_rate"]

    # optimizer = optim.Adam(policy_net.parameters())
    optimizer = optim.SGD(policy_net.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer"])

    checkpoint.pop("policy_net")
    checkpoint.pop("target_net")
    checkpoint.pop("optimizer")
    checkpoint.pop("i_episode")
    checkpoint.pop("learning_rate")

    return policy_net, target_net, optimizer, checkpoint
