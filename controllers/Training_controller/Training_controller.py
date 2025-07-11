import sys
import os
import numpy as np
import time
import logging
from collections import defaultdict

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Penyesuaian sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
controllers_dir = os.path.join(project_root, "controllers")
if controllers_dir not in sys.path: sys.path.insert(0, controllers_dir)

from dqn_agent.dqn_agent import DQNAgent 

# Konfigurasi default
DEFAULT_NUM_EXPLORATION_EPISODES: int = 500
DEFAULT_NUM_EXPLOITATION_EPISODES: int = 250
DEFAULT_NUM_ADDITIONAL_EPISODES: int = 150
DEFAULT_MAX_STEPS_PER_EPISODE: int = 1000

def train_agent(agent: DQNAgent, env,
                num_episodes_exploration: int = DEFAULT_NUM_EXPLORATION_EPISODES,
                num_episodes_exploitation: int = DEFAULT_NUM_EXPLOITATION_EPISODES,
                num_episodes_additional: int = DEFAULT_NUM_ADDITIONAL_EPISODES,
                max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE) -> list[float]:
    """
    Melatih agen DQN dalam lingkungan yang diberikan dengan logging yang efisien.
    """
    num_episodes_total = num_episodes_exploration + num_episodes_exploitation + num_episodes_additional
    logging.info(f"Starting training for {num_episodes_total} episodes.")
    
    episode_rewards: list[float] = []

    for episode in range(num_episodes_total):
        state = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        
        episode_losses, episode_q_values = [], []
        episode_reward_components = defaultdict(float)
        termination_reason = "Max Steps"

        while not done and step_count < max_steps_per_episode:
            action = agent.act(state, episode=episode)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1

            if 'reward_breakdown' in info:
                for key, value in info['reward_breakdown'].items():
                    episode_reward_components[key] += value

            if agent.memory.get_size() > agent.batch_size:
                replay_metrics = agent.replay()
                if replay_metrics:
                    loss, avg_max_q = replay_metrics
                    episode_losses.append(loss)
                    episode_q_values.append(avg_max_q)
            
            if done:
                termination_reason = info.get('reason', 'Unknown')
                break
        
        if (episode + 1) % agent.target_model_update_freq == 0:
            agent.update_target_model()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = agent.epsilon_min + (agent.epsilon_start - agent.epsilon_min) * \
                            np.exp(-agent.epsilon_decay_rate * (episode + 1))
            
        episode_rewards.append(total_reward)

        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_max_q = np.mean(episode_q_values) if episode_q_values else 0.0
        
        reward_breakdown_str = f"Track: {episode_reward_components['track']:.2f}, " \
                               f"Action: {episode_reward_components['action']:.2f}, " \
                               f"Obstacle: {episode_reward_components['obstacle']:.2f}"

        log_message = (f"Episode {episode+1}/{num_episodes_total} | "
                       f"Reward: {total_reward:.2f} | Steps: {step_count} | Epsilon: {agent.epsilon:.4f} | "
                       f"Avg Loss: {avg_loss:.4f} | Avg Max-Q: {avg_max_q:.2f} | "
                       f"Rewards ({reward_breakdown_str}) | "
                       f"Terminated by: {termination_reason}")
        logging.info(log_message)

    if agent.model_save_path:
        try:
            agent.save_model(agent.model_save_path)
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
    else:
        logging.warning("Model save path not set. Model will not be saved.")
    return episode_rewards

def evaluate_agent(agent: DQNAgent, env, num_eval_episodes: int = 5):
    logging.info(f"\nStarting evaluation for {num_eval_episodes} episodes.")
    if not (agent.model_save_path and os.path.exists(agent.model_save_path)):
        logging.warning(f"Model file not found at {agent.model_save_path}. Evaluation may be untrained.")
    else:
        try:
            agent.load_model(agent.model_save_path)
        except Exception as e:
            logging.error(f"Could not load model for evaluation: {e}")
            return
        
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    total_rewards: list[float] = []

    for episode in range(num_eval_episodes):
        state, done, episode_reward, step_count = env.reset(), False, 0.0, 0
        while not done and step_count < (DEFAULT_MAX_STEPS_PER_EPISODE * 2):
            action = agent.act(state, evaluate_mode=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            step_count += 1
        logging.info(f"Evaluation Episode {episode+1}/{num_eval_episodes} | Total Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
    
    logging.info(f"\nEvaluation complete. Average reward: {np.mean(total_rewards):.2f}")
    agent.epsilon = original_epsilon

if __name__ == "__main__":
    logging.warning("This module is typically called from E_Puck_controller.py in Webots.")