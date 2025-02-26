import torch
import torch.nn as nn
import numpy as np

class HybridACOTD3(nn.Module):
    def __init__(self, state_dim, action_dim, num_robots):
        super().__init__()
        self.actors = nn.ModuleList([
            ActorNetwork(state_dim, action_dim) 
            for _ in range(num_robots)
        ])
        self.critic = CentralizedCritic(state_dim*num_robots, action_dim*num_robots)
        self.aco = AdaptiveACO()
        self.replay_buffer = SharedReplayBuffer(1e6)
        
    def forward(self, states):
        pheromones = self.aco.get_pheromone_map()
        augmented_states = self._augment_states(states, pheromones)
        return [actor(state) for actor, state in zip(self.actors, augmented_states)]

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 64, 256),  # 64 = pheromone features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state) * torch.tensor([2.0, 1.0])  # Max velocities

class AdaptiveACO:
    def update_pheromones(self, paths, collisions):
        # Dynamic evaporation based on collision rate
        evaporation = 0.95 - 0.25 * np.clip(collisions, 0, 1)
        self.pheromones *= evaporation
        # Enhanced pheromone deposition
        for path in paths:
            self.pheromones[path] += 10 / (1 + len(path))
