import gymnasium as gym  # Updated import
from gymnasium import spaces  # Updated import
import numpy as np

class BandwidthOptimizationEnv(gym.Env):  # Inheriting from gymnasium.Env
    def __init__(self):
        super(BandwidthOptimizationEnv, self).__init__()
        
        # Define action and observation space
        # Actions: Increase, Decrease, or Maintain bandwidth
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [current bandwidth, user demand]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Initial state
        self.state = [0.5, 0.5]  # [current_bandwidth, user_demand]
        
        self.max_bandwidth = 1.0
        self.min_bandwidth = 0.0
        self.seed_val = None
        
    def step(self, action):
        current_bandwidth, user_demand = self.state
        
        # Action 0: decrease bandwidth, Action 1: maintain, Action 2: increase
        if action == 0:
            current_bandwidth = max(self.min_bandwidth, current_bandwidth - 0.1)
        elif action == 2:
            current_bandwidth = min(self.max_bandwidth, current_bandwidth + 0.1)
        
        # Simulate network performance (reward)
        performance = min(current_bandwidth, user_demand)
        
        # Reward is the performance (throughput), penalize if bandwidth is too high
        reward = performance - abs(current_bandwidth - user_demand)
        
        # Update user demand randomly (simulate varying network conditions)
        user_demand = np.random.uniform(0, 1)
        
        # Set the new state
        self.state = [current_bandwidth, user_demand]
        
        # Done after 100 steps (episodes)
        terminated = False  # Episode termination logic, update this based on your termination criteria
        truncated = False   # No truncation condition in this case, but you can set your own criteria
        
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # Set seed for reproducibility
        super().reset(seed=seed)
        self.seed_val = seed
        np.random.seed(self.seed_val)
        
        # Reset the state to initial condition
        self.state = [0.5, np.random.uniform(0, 1)]
        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode='human'):
        print(f"Current bandwidth: {self.state[0]}, User demand: {self.state[1]}")
