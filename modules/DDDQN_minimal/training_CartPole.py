import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import os
from utils import ReplayMemory, plot_durations
from model_DDDQN import DQN, select_action, optimize_model
from save_and_load import save_checkpoint, load_checkpoint

#######################  Parameters  ##############################

# Environment parameter
env_name = 'CartPole-v1'
is_unwrapped = False

# Model hyperparameters
input_size = 4      # Size of state
output_size = 2     # Number of discrete actions
ckpt_dir = "DDDQN_SGD_CartPoleV1_obs_checkpoints/"
save_ckpt_interval = 100

# Training parameters
# num_episodes = 1000
i_episode = 0      # This would determine which checkpoint to load, if the checkpoint exists
batch_size = 128
replaybuffer_size = 10000
learning_rate = 0.00003
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
target_update = 10


###################################################################


# Turn on pyplot's interactive mode
# VERY IMPORTANT because otherwise training stats plot will hault
plt.ion()

# Create OpenAI gym environment
env = gym.make(env_name)
if is_unwrapped:
    env = env.unwrapped

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current usable device is: ", device)

# Create the models
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Set up replay memory
memory = ReplayMemory(replaybuffer_size)

# Set up optimizer - Minimal
# optimizer = optim.Adam(policy_net.parameters())
optimizer = optim.SGD(policy_net.parameters(), lr=learning_rate)


###################################################################
# Start training

# Dictionary for extra training information to save to checkpoints
training_info = {"memory" : memory,
                 "episode reward" : [],
                 "training loss" : [],
                 "episode loss" : [],
                 "max reward achieved": 0,
                 "past 100 episodes mean reward": 0,
                 "max TD loss recorded" : 0,
                 "max episode loss recorded": 0}

while True:
    # Every save_ckpt_interval, Check if there is any checkpoint.
    # If there is, load checkpoint and continue training
    # Need to specify the i_episode of the checkpoint intended to load
    if i_episode % save_ckpt_interval == 0 and os.path.isfile(os.path.join(ckpt_dir, "ckpt_eps%d.pt" % i_episode)):
        policy_net, target_net, optimizer, training_info = load_checkpoint(ckpt_dir, i_episode, input_size, output_size, device=device)

    # Initialize the environment and state
    observation = env.reset()
    current_state = torch.tensor([observation], device=device, dtype=torch.float32)

    running_reward = 0
    running_minibatch_loss = 0
    running_episode_loss = 0
    for t in count():
        # Select and perform an action
        # Turn policy_net into evaluation mode to select an action given a single state
        policy_net.eval()

        action = select_action(current_state, policy_net, EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY ,
                               device=device)
        observation, reward, done, _ = env.step(action.item())
        env.render()

        # record reward
        running_reward += reward
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = torch.tensor([observation], device=device, dtype=torch.float32)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(current_state, action, next_state, reward)
        training_info["memory"] = memory

        # Compute the TD loss of current transition and store it into episode loss
        if not done:
            current_q = policy_net(current_state)[:, action].squeeze()
            target_q = policy_net(next_state).max() + reward.squeeze()
            target_q = torch.tensor(target_q.item(), device=device)
            trans_loss = F.smooth_l1_loss(current_q, target_q).item()
            # Record the TD loss
            running_episode_loss += trans_loss
            if trans_loss > training_info["max TD loss recorded"]:
                training_info["max TD loss recorded"] = trans_loss


        # Move to the next state
        current_state = next_state

        # Turn policy_net back to training mode to optimize on a batch of experiences
        policy_net.train()

        # Perform one step of the optimization (on the target network) and record the loss value
        minibatch_loss = optimize_model(batch_size, memory, policy_net, target_net, optimizer,
                              GAMMA=GAMMA, device=device)

        # update training loss log
        if minibatch_loss is not None:
            running_minibatch_loss += minibatch_loss

        if done:
            # Save and print episode stats (duration and episode loss)
            training_info["episode reward"].append(running_reward)
            if running_reward > training_info["max reward achieved"]:
                training_info["max reward achieved"] = running_reward
            training_info["past 100 episodes mean reward"] = \
                (sum(training_info["episode reward"][-100:]) / 100) if len(training_info["episode reward"])>=100 else 0
            training_info["training loss"].append(running_minibatch_loss / (t + 1))
            training_info["episode loss"].append(running_episode_loss / t)
            if (running_episode_loss / t) > training_info["max episode loss recorded"]:
                training_info["max episode loss recorded"] = running_episode_loss / t

            # Plot stats
            plot_durations(training_info["episode reward"],
                           training_info["training loss"],
                           training_info["episode loss"])

            print("=============  Episode: %d  =============" % (i_episode + 1))
            print("Episode reward: %d" % training_info["episode reward"][-1])
            print("Episode duration: %d" % (t + 1))
            print("Training loss: %f" % training_info["training loss"][-1])
            print("Episode loss: %f \n" % training_info["episode loss"][-1])
            print("Max reward achieved: %f" %  training_info["max reward achieved"])
            print("Max TD loss recorded: %f" % training_info["max TD loss recorded"])
            print("Max episode loss recorded: %f" % training_info["max episode loss recorded"])
            print("Past 100 episodes avg reward: %f \n\n" % training_info["past 100 episodes mean reward"])

            # Check if the problem is solved
            #  CartPole standard: average reward for the past 100 episode above 195
            if training_info["past 100 episodes mean reward"] > 195:
                print("\n\n\t Problem Solved !!!\n\n\n")

            break
    i_episode += 1

    # Update the target network, copying all weights and biases in DQN
    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Every save_ckpt_interval, save a checkpoint according to current i_episode.
    # Note that we use i_episode + 1
    if (i_episode + 1) % save_ckpt_interval == 0:
        save_checkpoint(ckpt_dir, policy_net, target_net, optimizer, i_episode + 1, learning_rate=learning_rate,
                        **training_info)
