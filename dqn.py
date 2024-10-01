import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from main import BandwidthOptimizationEnv

# Directory to save the model
model_dir = "saved_models"
model_filename = "dqn_bandwidth_optimization.zip"
model_path = os.path.join(model_dir, model_filename)

# Create the environment
env = BandwidthOptimizationEnv()

# Check if the environment follows the Gym API
check_env(env, warn=True)

# Create the DQN model
model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3)

# Train the model
model.learn(total_timesteps=10000)

# Create directory if it does not exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
model.save(model_path)
print(f"Model saved at: {model_path}")

# Load the model for evaluation
if os.path.exists(model_path):
    model = DQN.load(model_path)
    print(f"Model loaded from: {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Evaluate the trained agent
state, _ = env.reset()  # Extract only the observation from the reset tuple
for step in range(1000):
    action, _states = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    # If the episode is finished, reset the environment
    if terminated or truncated:
        state, _ = env.reset() 
