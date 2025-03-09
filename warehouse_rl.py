import os
# Workaround for OpenMP duplicate runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import math

# Setting PyTorch thread settings for performance tuning
torch.set_num_threads(4)  # Adjust based on your CPU

# Setting device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Warehouse Environment using OpenAI Gym
class WarehouseEnv(gym.Env):
    def __init__(self):
        super(WarehouseEnv, self).__init__()
        self.warehouse_size = 10.0  # 10x10 area

        # Defining action space: continuous actions in [-1,1] for linear and angular velocity
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                           high=np.array([1.0, 1.0], dtype=np.float32),
                                           dtype=np.float32)
        # Defining observation space: state vector of 8 values
        # [x, y, theta, carrying, pkg_x, pkg_y, deliv_x, deliv_y]
        high = np.array([self.warehouse_size, self.warehouse_size, np.pi, 1.0,
                         self.warehouse_size, self.warehouse_size, self.warehouse_size, self.warehouse_size],
                        dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        # Simulation parameters
        self.dt = 0.1  # time step
        self.robot_radius = 0.2
        self.package_radius = 0.2
        self.delivery_radius = 0.2

        # Defining obstacles as circles
        self.obstacles = [
            (3.0, 3.0, 0.5),
            (7.0, 2.0, 0.5),
            (5.0, 7.0, 0.5),
            (2.0, 8.0, 0.5)
        ]

    def reset(self):
        # Random initial robot position and orientation
        self.robot_x = np.random.uniform(0.5, self.warehouse_size - 0.5)
        self.robot_y = np.random.uniform(0.5, self.warehouse_size - 0.5)
        self.robot_theta = np.random.uniform(-np.pi, np.pi)
        self.carrying = 0  # Not carrying package initially

        # Random package location
        self.pkg_x = np.random.uniform(0.5, self.warehouse_size - 0.5)
        self.pkg_y = np.random.uniform(0.5, self.warehouse_size - 0.5)

        # Random delivery location
        while True:
            self.deliv_x = np.random.uniform(0.5, self.warehouse_size - 0.5)
            self.deliv_y = np.random.uniform(0.5, self.warehouse_size - 0.5)
            if np.linalg.norm(np.array([self.pkg_x, self.pkg_y]) - np.array([self.deliv_x, self.deliv_y])) > 3.0:
                break

        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([self.robot_x, self.robot_y, self.robot_theta, float(self.carrying),
                         self.pkg_x, self.pkg_y, self.deliv_x, self.deliv_y], dtype=np.float32)

    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        linear_vel, angular_vel = action

        # Update robot state using basic kinematics
        self.robot_theta += angular_vel * self.dt
        self.robot_x += linear_vel * np.cos(self.robot_theta) * self.dt
        self.robot_y += linear_vel * np.sin(self.robot_theta) * self.dt

        # Keep the robot within warehouse limits
        self.robot_x = np.clip(self.robot_x, 0, self.warehouse_size)
        self.robot_y = np.clip(self.robot_y, 0, self.warehouse_size)

        reward = -0.1  # Small step penalty to encourage faster task completion

        # Check for collision with obstacles
        for (ox, oy, oradius) in self.obstacles:
            if np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([ox, oy])) <= (self.robot_radius + oradius):
                reward -= 10.0  # Collision penalty
                self.done = True
                return self._get_state(), reward, self.done, {"info": "Collision with obstacle"}

        # Check for package pickup
        if self.carrying == 0 and np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([self.pkg_x, self.pkg_y])) <= (self.robot_radius + self.package_radius):
            self.carrying = 1
            reward += 5.0  # Reward for picking up the package

        # Check for successful delivery
        if self.carrying == 1 and np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([self.deliv_x, self.deliv_y])) <= self.delivery_radius:
            reward += 10.0  # Delivery reward
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self, mode='human'):
        # Simple visualization using matplotlib
        plt.clf()
        plt.xlim(0, self.warehouse_size)
        plt.ylim(0, self.warehouse_size)
        # Drawing obstacles
        for (ox, oy, oradius) in self.obstacles:
            circle = plt.Circle((ox, oy), oradius, color='red')
            plt.gca().add_patch(circle)
        # Drawing package
        if self.carrying == 0:
            pkg = plt.Circle((self.pkg_x, self.pkg_y), self.package_radius, color='green')
            plt.gca().add_patch(pkg)
        # Drawing delivery location
        deliv = plt.Circle((self.deliv_x, self.deliv_y), self.delivery_radius, color='blue', fill=False, linestyle='--')
        plt.gca().add_patch(deliv)
        # Drawing robot
        robot = plt.Circle((self.robot_x, self.robot_y), self.robot_radius, color='black')
        plt.gca().add_patch(robot)
        plt.pause(0.001)
        plt.draw()

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# TD3 Agent Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# TD3 Algorithm Implementation
class TD3:
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        # Sample a batch of transitions
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # Select action according to policy with added noise
        noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = self.actor_target(next_state)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-values
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks with Polyak averaging
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ACO for Path Planning
def aco_path_planner(start, goal, grid, num_ants=10, num_iterations=50, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    rows, cols = grid.shape
    pheromone = np.ones((rows, cols))

    def get_neighbors(cell):
        x, y = cell
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    best_path = None
    best_length = float('inf')

    for iteration in range(num_iterations):
        all_paths = []
        for ant in range(num_ants):
            path = [start]
            visited = set()
            visited.add(start)
            current = start
            while current != goal:
                neighbors = get_neighbors(current)
                if not neighbors:
                    break
                # Computing probability for each neighbour
                probs = []
                for n in neighbors:
                    if n in visited:
                        probs.append(0.0)
                    else:
                        distance = np.linalg.norm(np.array(n) - np.array(goal))
                        heuristic = 1.0 / (distance + 1e-6)
                        probs.append((pheromone[n] ** alpha) * (heuristic ** beta))
                sum_probs = sum(probs)
                if sum_probs == 0:
                    break
                probs = [p / sum_probs for p in probs]
                next_cell = random.choices(neighbors, weights=probs, k=1)[0]
                path.append(next_cell)
                visited.add(next_cell)
                current = next_cell
                # To avoid infinite loops
                if len(path) > rows * cols:
                    break
            if current == goal:
                all_paths.append(path)
                if len(path) < best_length:
                    best_path = path
                    best_length = len(path)
        # Pheromone evaporation and deposition
        pheromone *= (1 - evaporation_rate)
        for path in all_paths:
            for cell in path:
                pheromone[cell] += 1.0 / len(path)
    return best_path

def visualize_aco_path(grid, path):
    plt.imshow(grid, cmap='gray_r')
    if path:
        xs = [p[1] for p in path]  # columns
        ys = [p[0] for p in path]  # rows
        plt.plot(xs, ys, marker='o', color='red')
    plt.title("ACO Path Planning")
    plt.show()

# Main Training Loop & Testing
def main():
    # Initialize environment and agent
    env = WarehouseEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0  # actions are in [-1,1]

    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(capacity=1000000)

    max_episodes = 100
    max_steps = 200
    batch_size = 100
    exploration_noise = 0.1

    rewards_history = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            # Select action and add exploration noise
            action = agent.select_action(state)
            action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                break

            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size)

        rewards_history.append(episode_reward)
        print(f"Episode: {episode}, Reward: {episode_reward}")

        # Visualizing the environment every 10 episodes
        if episode % 10 == 0:
            env.render()

    # Plotting training reward history
    plt.figure()
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward History")
    plt.show()

    # Testing the trained agent
    test_episodes = 5
    for episode in range(test_episodes):
        state = env.reset()
        env.render()
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break
        print(f"Test Episode: {episode}, Total Reward: {total_reward}")

# ACO Path Planning Example
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    # Placing some obstacles
    grid[3, 3] = 1
    grid[7, 2] = 1
    grid[5, 7] = 1
    grid[2, 8] = 1
    start = (0, 0)
    goal = (9, 9)
    best_path = aco_path_planner(start, goal, grid)
    print("Best path found by ACO:", best_path)
    visualize_aco_path(grid, best_path)

if __name__ == "__main__":
    main()
