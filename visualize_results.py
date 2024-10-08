import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from main import BandwidthOptimizationEnv

# Load the trained model
model_dir = "saved_models"
model_filename = "dqn_bandwidth_optimization.zip"
model_path = os.path.join(model_dir, model_filename)

if os.path.exists(model_path):
    model = DQN.load(model_path)
    print(f"Model loaded from: {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Create the environment
env = BandwidthOptimizationEnv()

# Run the model and collect data
num_episodes = 2
max_steps = 5


all_bandwidths = []
all_demands = []
all_rewards = []
all_efficiencies = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_bandwidths = []
    episode_demands = []
    episode_rewards = []
    episode_efficiencies = []
    
    for step in range(max_steps):
        action, _states = model.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        
        bandwidth, demand = state
        episode_bandwidths.append(bandwidth)
        episode_demands.append(demand)
        episode_rewards.append(reward)
        
        # Calculate efficiency
        performance = min(bandwidth, demand)
        efficiency = performance / bandwidth if bandwidth > 0 else 0
        episode_efficiencies.append(efficiency)
        
        if terminated or truncated:
            break
    
    all_bandwidths.append(episode_bandwidths)
    all_demands.append(episode_demands)
    all_rewards.append(episode_rewards)
    all_efficiencies.append(episode_efficiencies)
    
    # Print average efficiency for this episode
    avg_efficiency = np.mean(episode_efficiencies)
    print(f"Episode {episode + 1} - Average Efficiency: {avg_efficiency:.2f}")

# Calculate overall average efficiency
overall_avg_efficiency = np.mean([np.mean(eff) for eff in all_efficiencies])
print(f"\nOverall Average Efficiency: {overall_avg_efficiency:.2f}")

# Plotting
plt.figure(figsize=(20, 5))

# Plot bandwidth and user demand
plt.subplot(1, 3, 1)
for i in range(num_episodes):
    plt.plot(all_bandwidths[i], label=f'Bandwidth (Episode {i+1})')
    plt.plot(all_demands[i], label=f'User Demand (Episode {i+1})', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Bandwidth and User Demand over Time')
plt.legend()

# Plot rewards
plt.subplot(1, 3, 2)
for i in range(num_episodes):
    plt.plot(all_rewards[i], label=f'Episode {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Rewards over Time')
plt.legend()

# Plot efficiencies
plt.subplot(1, 3, 3)
for i in range(num_episodes):
    plt.plot(all_efficiencies[i], label=f'Episode {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Efficiency')
plt.title('Efficiency over Time')
plt.legend()

plt.tight_layout()
plt.savefig('bandwidth_optimization_results.png')
plt.show()

print("Visualization saved as 'bandwidth_optimization_results.png'")