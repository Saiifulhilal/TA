## Training_controller.py (Versi Perbaikan)

import sys
import os
import numpy as np
import logging
from collections import defaultdict

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path adjustments (sudah benar, tidak ada perubahan) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)
controllers_dir = os.path.join(project_root, "controllers")
if controllers_dir not in sys.path: sys.path.insert(0, controllers_dir)

# Impor kelas DQNAgent
from dqn_agent.dqn_agent import DQNAgent

# --- Konstanta (tidak ada perubahan) ---
DEFAULT_NUM_EXPLORATION_EPISODES: int = 500
DEFAULT_NUM_EXPLOITATION_EPISODES: int = 250
DEFAULT_NUM_ADDITIONAL_EPISODES: int = 150
DEFAULT_MAX_STEPS_PER_EPISODE: int = 1000

def train_agent(agent: DQNAgent, env,
                num_episodes_exploration: int = DEFAULT_NUM_EXPLORATION_EPISODES,
                num_episodes_exploitation: int = DEFAULT_NUM_EXPLOITATION_EPISODES,
                num_episodes_additional: int = DEFAULT_NUM_ADDITIONAL_EPISODES,
                max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
                model_save_path: str = None) -> list[float]:
    """Melatih agen DQN dalam lingkungan yang diberikan dengan logging."""
    num_episodes_total = num_episodes_exploration + num_episodes_exploitation + num_episodes_additional
    logging.info(f"Memulai training untuk {num_episodes_total} episode.")
    
    # Tetapkan path penyimpanan model pada agen
    if model_save_path:
        agent.set_save_path(model_save_path)

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
            # PERBAIKAN: Panggilan ke agent.act() disederhanakan.
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1

            if 'reward_breakdown' in info:
                for key, value in info['reward_breakdown'].items():
                    episode_reward_components[key] += value

            # Gunakan len(agent.memory) yang lebih Pythonic
            if len(agent.memory) > agent.batch_size:
                replay_metrics = agent.replay()
                if replay_metrics:
                    loss, avg_max_q = replay_metrics # Membutuhkan replay() untuk mengembalikan 2 nilai
                    episode_losses.append(loss)
                    episode_q_values.append(avg_max_q)
            
            if done:
                termination_reason = info.get('reason', 'Goal Reached')
                break
        
        # PERBAIKAN: Logika pembaruan target model dan epsilon dipindahkan ke dalam agen
        # untuk enkapsulasi yang lebih baik.
        agent.update_target_model_if_needed(episode)
        agent.update_epsilon(episode) # Memperbarui epsilon berdasarkan nomor episode

        episode_rewards.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_max_q = np.mean(episode_q_values) if episode_q_values else 0.0
        
        reward_breakdown_str = (f"Track: {episode_reward_components['track']:.2f}, "
                                f"Action: {episode_reward_components['action']:.2f}, "
                                f"Obstacle: {episode_reward_components['obstacle']:.2f}")

        log_message = (f"Episode {episode+1}/{num_episodes_total} | "
                       f"Reward: {total_reward:.2f} | Steps: {step_count} | Epsilon: {agent.epsilon:.4f} | "
                       f"Avg Loss: {avg_loss:.4f} | Avg Max-Q: {avg_max_q:.2f} | "
                       f"Rewards ({reward_breakdown_str}) | "
                       f"Terminated by: {termination_reason}")
        logging.info(log_message)

    agent.save_model() # Menyimpan model di akhir training
    return episode_rewards


def evaluate_agent(agent: DQNAgent, env, num_eval_episodes: int = 5):
    logging.info(f"\nMemulai evaluasi untuk {num_eval_episodes} episode.")
    
    try:
        agent.load_model() # Memuat model dari path yang sudah tersimpan di agen
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Tidak dapat memuat model untuk evaluasi: {e}")
        return
        
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0 # Mode evaluasi murni (tanpa eksplorasi)
    total_rewards: list[float] = []

    for episode in range(num_eval_episodes):
        state, done, episode_reward, step_count = env.reset(), False, 0.0, 0
        while not done and step_count < (DEFAULT_MAX_STEPS_PER_EPISODE * 2):
            # PERBAIKAN: Panggilan ke agent.act() disederhanakan.
            action = agent.act(state)
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            step_count += 1
        logging.info(f"Evaluasi Episode {episode+1}/{num_eval_episodes} | Total Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
    
    logging.info(f"\nEvaluasi selesai. Rata-rata reward: {np.mean(total_rewards):.2f}")
    agent.epsilon = original_epsilon # Kembalikan epsilon ke nilai semula


if __name__ == "__main__":
    logging.warning("Modul ini biasanya dipanggil dari E_Puck_controller.py di Webots.")