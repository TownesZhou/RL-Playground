import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std=0.0):
        super(ActorCritic, self).__init__()
        # action range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),
            nn.Linear(n_var, n_var),
            nn.Tanh(),
            nn.Linear(n_var, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)
        self.action_dim = action_dim

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)

        # dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))
        # action = dist.sample()
        # action_logprob = dist.log_prob(action)

        m = []
        action = None
        log_prob = None
        for i in range(self.action_dim):
            m.append(Normal(action_mean[0][i], self.action_var[i]))
            if action is None:
                action = m[-1].sample().unsqueeze(0)
                log_prob = m[-1].log_prob(action)
            else:
                action = torch.cat([action, m[-1].sample().unsqueeze(0)])
                log_prob += m[-1].log_prob(action[-1])

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list in tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    env_name = "LunarLanderContinuous-v2"
    solved_reward = 200  # stop training if avg_reward > solved_reward
    max_ep = 10000
    max_timesteps = 1000
    render = False
    log_interval = 50  # print avg reward after n episodes
    n_latent_var = 64  # number of variables in hidden layer
    n_update = 5  # update policy every n episodes
    action_std = 0.1  # constant std for action distribution
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    running_reward = 0
    avg_length = 0
    for i_episode in range(1, max_ep + 1):
        state = env.reset()
        for t in range(max_timesteps):
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward:
            memory.rewards.append(reward)

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # update after n episodes
        if i_episode % n_update:
            ppo.update(memory)
            memory.clear_memory()

        # log
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './LunarLander_{}_{}_{}.pth'.format(
                lr, betas[0], betas[1]))
            break

        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
