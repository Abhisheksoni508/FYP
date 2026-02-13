"""It is actually okay that it is empty right now because your main_train_rl.py is likely importing DQN directly from stable_baselines3. 
The file was probably created as a placeholder for modularity but never used."""


from stable_baselines3 import DQN
import os
from src.config import *

def create_dqn_agent(env):
    """
    Initializes the DQN agent with hyperparameters defined in config.py.
    
    Args:
        env: The Gymnasium environment (PdMEnvironment)
        
    Returns:
        model: An untrained DQN model
    """
    model = DQN(
        policy="MlpPolicy",         # Multi-Layer Perceptron (Standard Neural Net)
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,    # Experience Replay buffer
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,                # Discount factor
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/"
    )
    return model

def load_saved_agent(path, env=None):
    """
    Loads a trained agent from disk.
    
    Args:
        path (str): Path to the .zip file
        env (gym.Env, optional): Environment to attach (needed for retraining)
        
    Returns:
        model: The loaded DQN model
    """
    # Standardize path handling
    if not path.endswith(".zip"):
        clean_path = path
    else:
        clean_path = path.replace(".zip", "")
        
    if not os.path.exists(clean_path + ".zip"):
        print(f"Warning: Model not found at {clean_path}")
        return None
        
    print(f"Loading agent from {clean_path}...")
    model = DQN.load(clean_path, env=env)
    return model