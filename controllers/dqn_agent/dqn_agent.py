import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
import logging
import os

# Konfigurasi logging dasar untuk output yang bersih
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReplayBuffer:
    """Buffer untuk menyimpan dan mengambil sampel pengalaman (experiences) untuk training."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Menambahkan satu set pengalaman ke dalam buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mengambil sampel acak sebanyak batch_size dari buffer."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Replay buffer hanya memiliki {len(self.buffer)} sampel, butuh {batch_size}.")
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones.astype(bool)

    def __len__(self) -> int:
        """Mengembalikan ukuran buffer saat ini. Memungkinkan `len(replay_buffer)`."""
        return len(self.buffer)

class DQNAgent:
    """Implementasi agen Deep Q-Network (DQN) yang telah disempurnakan."""
    def __init__(self, state_size, action_size, learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_decay_rate: float = 0.005, epsilon_min: float = 0.01,
                 batch_size: int = 64, memory_size: int = 10000, target_model_update_freq: int = 10):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.target_model_update_freq = target_model_update_freq
        self.model_save_path = None

    def _build_model(self):
        """Membangun model Q-network, beradaptasi untuk input gambar atau numerik."""
        if isinstance(self.state_size, tuple):
            inputs = keras.layers.Input(shape=self.state_size)
            x = keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
            x = keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
            x = keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(512, activation='relu')(x)
            outputs = keras.layers.Dense(self.action_size, activation='linear')(x)
        else:
            inputs = keras.layers.Input(shape=(self.state_size,))
            x = keras.layers.Dense(64, activation='relu')(inputs)
            x = keras.layers.Dense(64, activation='relu')(x)
            outputs = keras.layers.Dense(self.action_size, activation='linear')(x)
            
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # --- PERUBAHAN DI SINI ---
        # Menggunakan objek fungsi loss secara eksplisit untuk menghindari error saat memuat model
        loss_function = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss_function, optimizer=optimizer)
        # --- AKHIR PERUBAHAN ---

        return model
        
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)
        
    def act(self, state: np.ndarray) -> int:
        if random.uniform(0, 1) > self.epsilon:
            state_tensor = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state_tensor, verbose=0)[0]
            return np.argmax(q_values)
        else:
            return random.randrange(self.action_size)

    def replay(self) -> tuple[float, float] | None:
        """Melatih model dan mengembalikan loss serta rata-rata max Q-value."""
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        q_current = self.model.predict(states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        
        avg_max_q = np.mean(np.max(q_current, axis=1))

        max_q_next = np.amax(q_next_target, axis=1)
        target_values = rewards + self.gamma * max_q_next * (1 - dones)
        
        targets = np.copy(q_current)
        batch_indices = np.arange(self.batch_size)
        targets[batch_indices, actions.astype(int)] = target_values
            
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        return loss, avg_max_q

    def update_epsilon(self, episode: int):
        """Memperbarui epsilon menggunakan formula exponential decay berdasarkan nomor episode."""
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                           np.exp(-self.epsilon_decay_rate * episode)

    def update_target_model(self):
        """Menyalin bobot dari model utama ke target model."""
        self.target_model.set_weights(self.model.get_weights())

    def update_target_model_if_needed(self, episode: int):
        """Menyalin bobot ke target model pada frekuensi yang ditentukan."""
        if (episode + 1) % self.target_model_update_freq == 0:
            self.update_target_model()
            logging.info(f"Target model diperbarui pada episode {episode + 1}.")

    def set_save_path(self, path: str):
        """Menetapkan path untuk menyimpan/memuat model."""
        self.model_save_path = path

    def save_model(self):
        """Menyimpan model Q-network ke path yang sudah ditentukan."""
        if not self.model_save_path:
            logging.warning("Path penyimpanan model tidak diatur. Model tidak akan disimpan.")
            return
        try:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            self.model.save(self.model_save_path)
            logging.info(f"Model berhasil disimpan di: {self.model_save_path}")
        except Exception as e:
            logging.error(f"Gagal menyimpan model ke {self.model_save_path}: {e}")

    def load_model(self):
        """Memuat model dari path yang sudah ditentukan."""
        if not self.model_save_path or not os.path.exists(self.model_save_path):
            logging.error(f"File model tidak ditemukan di {self.model_save_path}")
            raise FileNotFoundError(f"File model tidak ditemukan di {self.model_save_path}")
        
        try:
            self.model = keras.models.load_model(self.model_save_path)
            self.update_target_model()
            logging.info(f"Model berhasil dimuat dari: {self.model_save_path}")
        except Exception as e:
            logging.error(f"Gagal memuat model dari {self.model_save_path}: {e}")
            raise