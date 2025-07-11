import numpy as np
import random
from collections import deque
from tensorflow import keras
import tensorflow as tf
import logging
import os

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Kelas ReplayBuffer ---
class ReplayBuffer:
    """
    Buffer untuk menyimpan dan mengambil sampel pengalaman pelatihan agen.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError(f"Replay buffer has only {len(self.buffer)} samples, but {batch_size} requested.")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def get_size(self) -> int:
        return len(self.buffer)

# --- Kelas DQNAgent ---
class DQNAgent:
    """
    Implementasi agen Deep Q-Network (DQN) untuk reinforcement learning.
    """
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_decay_rate: float = 0.005, epsilon_min: float = 0.01,
                 batch_size: int = 32, memory_size: int = 2000, target_model_update_freq: int = 10):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() 

        self.target_model_update_freq = target_model_update_freq 
        self.model_save_path = None

    def _build_model(self) -> keras.Sequential:
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(24, activation='relu'), 
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state: np.ndarray, episode: int = None, evaluate_mode: bool = False) -> int:
        if evaluate_mode or random.uniform(0, 1) > self.epsilon:
            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            return np.argmax(q_values)
        else:
            return random.randrange(self.action_size)

    def replay(self) -> tuple[float, float] | None:
        if self.memory.get_size() < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        q_current = self.model.predict(states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        
        avg_max_q = np.mean(np.max(q_current, axis=1))

        targets = np.copy(q_current)
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(q_next_target[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        
        loss = history.history['loss'][0]
        
        return loss, avg_max_q

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        logging.debug("Target model updated.")

    def save_model(self, filename: str):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.model.save(filename)
            logging.info(f"Model saved to: {filename}")
        except Exception as e:
            logging.error(f"Error saving model to {filename}: {e}")

    # --- FUNGSI INI TELAH DIPERBARUI ---
    def load_model(self, filename: str):
        """
        Memuat model dari file .h5 dengan penanganan custom objects.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found at {filename}")
        try:
            # Definisikan custom object untuk mengatasi error 'mse' saat memuat
            custom_obj = {'mse': tf.keras.losses.MeanSquaredError()}
            
            # Gunakan custom_objects saat memuat model
            self.model = keras.models.load_model(filename, custom_objects=custom_obj)
            self.target_model = keras.models.load_model(filename, custom_objects=custom_obj)
            
            self.update_target_model() 
            logging.info(f"Model loaded successfully from: {filename}")
        except Exception as e:
            logging.error(f"Error loading model from {filename}: {e}")
            raise