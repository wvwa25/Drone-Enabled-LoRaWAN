import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, env, input_dim, output_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 batch_size=32, learning_rate=1e-3, target_update_freq=100):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        # Initialize the Q-network and target network
        self.num_actions = 10  # Number of discrete actions per dimension
        total_actions = self.num_actions ** output_dim  # Calculate total number of discrete actions
        self.q_network = QNetwork(input_dim, total_actions).float()
        self.target_network = QNetwork(input_dim, total_actions).float()
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Store action space information
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high
        self.output_dim = output_dim

class DQNAgent:
    def __init__(self, env, input_dim, output_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 batch_size=32, learning_rate=1e-3, target_update_freq=100):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        # Initialize the Q-network and target network
        self.num_actions = 10  # Number of discrete actions per dimension
        total_actions = self.num_actions ** output_dim  # Calculate total number of discrete actions
        self.q_network = QNetwork(input_dim, total_actions).float()
        self.target_network = QNetwork(input_dim, total_actions).float()
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Store action space information
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high
        self.output_dim = output_dim

    def _discretize_action(self, action_idx):
        actions = []
        remaining_idx = action_idx
        for dim in range(self.output_dim):
            idx = remaining_idx % self.num_actions
            remaining_idx //= self.num_actions
            action = self.action_space_low[dim] + (idx / (self.num_actions - 1)) * (
                    self.action_space_high[dim] - self.action_space_low[dim])
            actions.append(action)
        return np.array(actions)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random discrete action
            total_actions = self.num_actions ** len(self.action_space_high)
            discrete_action = np.random.randint(0, total_actions)
            return self._discretize_action(discrete_action)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            discrete_action = torch.argmax(q_values).item()
            return self._discretize_action(discrete_action)

    def store_experience(self, state, action, reward, next_state, done):
        # Find closest discrete action index
        action_dim = len(self.action_space_high)
        discrete_action = 0
        for dim in range(action_dim):
            normalized_action = (action[dim] - self.action_space_low[dim]) / (
                    self.action_space_high[dim] - self.action_space_low[dim])
            idx = int(round(normalized_action * (self.num_actions - 1)))
            idx = max(0, min(idx, self.num_actions - 1))
            discrete_action = discrete_action * self.num_actions + idx

        self.replay_buffer.append((state, discrete_action, reward, next_state, done))

    def update_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        next_q_values = self.target_network(next_states)
        next_q_values = next_q_values.max(1)[0]

        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, num_episodes):
        rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                self.store_experience(state, action, reward, next_state, done)
                self.update_q_network()
                state = next_state

            self.update_epsilon()

            if episode % self.target_update_freq == 0:
                self.update_target_network()

            rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        return rewards