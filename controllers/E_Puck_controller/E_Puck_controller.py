import os
import time
import logging
from typing import List, Dict, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import dari Stable-Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# Import dari Webots
# Pastikan modul 'controller' tersedia di lingkungan Anda
try:
    from controller import Supervisor, Lidar
except ImportError:
    print("Peringatan: Modul 'controller' Webots tidak ditemukan. Skrip ini hanya akan berjalan di dalam Webots.")
    # Definisikan kelas placeholder agar tidak terjadi error saat di luar Webots
    class Supervisor:
        def getBasicTimeStep(self): return 32 # Dummy value
        def getSelf(self): return type('Node', (object,), {'getField': lambda s, f: type('Field', (object,), {'setSFVec3f': lambda s, v: None, 'setSFRotation': lambda s, r: None})()})()
        def step(self, timestep): return 0 # Dummy value
        def getWorldPath(self): return "dummy_world.wbt" # Dummy value
    class Lidar:
        def enable(self, timestep): pass
        def enablePointCloud(self): pass
        def getMaxRange(self): return 1.0 # Dummy value
        def getHorizontalResolution(self): return 0 # Dummy value
        def getPointCloud(self): return [] # Dummy value
    # Placeholder for Device objects
    class Device:
        def enable(self, timestep): pass
        def getValue(self): return 0.0 # Dummy value
    class Motor:
        def setPosition(self, pos): pass
        def setVelocity(self, vel): pass
    class DummyRobot: # A more complete dummy for Supervisor
        def getDevice(self, name):
            if 'motor' in name: return Motor()
            return Device()
        def getBasicTimeStep(self): return 32
        def getSelf(self): return type('Node', (object,), {'getField': lambda s, f: type('Field', (object,), {'setSFVec3f': lambda s, v: None, 'setSFRotation': lambda s, r: None})()})()
        def step(self, timestep): return 0
        def getWorldPath(self): return "dummy_world.wbt"
    Supervisor = DummyRobot # Assign the dummy robot to Supervisor for testing outside Webots


# ==============================================================================
# KONFIGURASI UTAMA
# ==============================================================================
# Sentralisasikan semua parameter yang dapat diubah di sini agar mudah dimodifikasi.
TRAINING_CONFIG = {
    "phase_name": "Pelatihan Fase 1.04",
    "training_cycle": 1,
    "track_id": 0,
    "models_dir": "saved_models_sb3",
    # model_filename ini adalah NAMA FILE UNTUK MENYIMPAN model hasil pelatihan ini
    "model_filename": "dqn_model_Fase1.04.zip", # <-- DIUBAH: Nama file untuk menyimpan model BARU
    "tensorboard_log_dir": "tensorboard_logs/",

    "training": {
        "num_episodes": 1000,
        "max_steps_per_episode": 400, # Ditingkatkan menjadi 200 langkah per episode
        "algorithm": "DQN",
    },

    "dqn_params": {
        "policy": "MlpPolicy",
        "learning_rate": 0.001,      # Learning rate standar untuk pembelajaran cepat
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,               # Fokus pada reward jangka panjang
        "train_freq": (1, "step"),
        "target_update_interval": 500,
        "verbose": 0,
        # --- Parameter untuk mengontrol Epsilon ---
        "exploration_initial_eps": 1.0,  # Nilai epsilon awal
        "exploration_final_eps": 0.05,   # Nilai epsilon akhir
        "exploration_fraction": 0.1,     # Fraksi dari total timesteps untuk penurunan epsilon
    },

    "env_params": {
        "max_lives": 3,
        "buffer_size": 5,
        "num_lidar_sectors": 8,
        "action_space_velocities": [(6.28, 6.28), (3.14, 6.28), (6.28, 3.14)], # Maju, Kiri, Kanan
        "checkpoints": {0: [-1.62948, 1.01949, -0.00963313]},
        
        # Thresholds (nilai 0-1)
        "line_detect_threshold": 0.5,
        "center_line_detect_threshold": 0.8,
        "proximity_avoid_threshold": 0.5,
        "proximity_collision_threshold": 0.7,

        # Rewards & Penalties (Diperbarui untuk Fase Easy)
        "reward_on_track_center": 5.0,      # Sangat memprioritaskan tengah jalur
        "reward_on_track_edge": 0.7,        # Reward untuk berada di pinggir jalur
        "reward_avoid_collision": 1.5,      # Turunkan sedikit, karena sedikit rintangan di fase easy
        "reward_action_straight": 0.5,      # Dorong agen untuk maju secara konsisten
        "penalty_time_step": -0.05,         # Penalty kecil untuk setiap langkah waktu
        "penalty_off_track": -6.0,          # Jadikan ini penalty terberat untuk keluar jalur
        "penalty_collision": -5.0,          # Penalty besar saat terjadi tabrakan
        "penalty_action_turn": -0.05,       # Penalty kecil untuk aksi berbelok
    }
}

# ==============================================================================
# KONSTANTA
# ==============================================================================
# Menggunakan konstanta untuk alasan terminasi agar konsisten dan menghindari typo.
TERMINATION_REASON_MAX_STEPS = "Max Steps"
TERMINATION_REASON_OFF_TRACK = "Off Track"
TERMINATION_REASON_NO_LIVES = "No Lives Left"

# ==============================================================================
# FUNGSI UTILITAS
# ==============================================================================
def setup_logging():
    """Mengatur konfigurasi logging dasar."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def format_duration(seconds: float) -> str:
    """Mengubah detik menjadi format jam, menit, detik yang mudah dibaca."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours} jam, {minutes} menit, {secs} detik"

# ==============================================================================
# KELAS CALLBACK UNTUK LOGGING KUSTOM
# ==============================================================================
class CustomLoggerCallback(BaseCallback):
    """
    Callback kustom untuk mencatat statistik episode secara detail dan
    menyediakan ringkasan di akhir fase pelatihan.
    """
    def __init__(self, config: Dict[str, Any], verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.phase_name = config["phase_name"]
        self.training_cycle = config["training_cycle"]
        self.num_episodes = config["training"]["num_episodes"]
        self.episode_count = 0
        self.summary = {
            TERMINATION_REASON_MAX_STEPS: 0,
            TERMINATION_REASON_OFF_TRACK: 0,
            TERMINATION_REASON_NO_LIVES: 0,
            "Collision Events": 0,
            "Total Lives Lost": 0,
            "Total Steps": 0
        }

    def _on_step(self) -> bool:
        """Dipanggil setelah setiap langkah di environment."""
        if self.locals['dones'][0]:
            self.episode_count += 1
            info = self.locals['infos'][0]

            if 'episode' in info:
                ep_rew = info['episode']['r']
                ep_len = info['episode']['l']
                term_reason = info.get('termination_reason', TERMINATION_REASON_MAX_STEPS)
                rew_breakdown = info.get('reward_breakdown', {})
                lives_lost = info.get('lives_lost', 0)
                collisions = info.get('collision_events', 0)

                self.summary[term_reason] = self.summary.get(term_reason, 0) + 1
                self.summary["Total Lives Lost"] += lives_lost
                self.summary["Collision Events"] += collisions

                # Mengambil nilai epsilon dari model
                epsilon = self.model.exploration_rate if hasattr(self.model, 'exploration_rate') else -1

                log_msg = (
                    f"Siklus {self.training_cycle} | {self.phase_name} | Ep {self.episode_count}/{self.num_episodes} | "
                    f"Reward: {ep_rew:.2f} | Steps: {ep_len} | Epsilon: {epsilon:.4f} | "
                    f"Terminasi: {term_reason}\n"
                    f"    Breakdown Reward -> "
                    f"Track: {rew_breakdown.get('track', 0.0):.2f}, "
                    f"Obstacle: {rew_breakdown.get('obstacle', 0.0):.2f}, "
                    f"Avoid: {rew_breakdown.get('avoid', 0.0):.2f}, "
                    f"Action: {rew_breakdown.get('action', 0.0):.2f}"
                )
                logging.info(log_msg)
        return True

    def _on_training_end(self) -> None:
        """Menyimpan total langkah yang telah dijalani di akhir pelatihan."""
        self.summary["Total Steps"] = self.num_timesteps

# ==============================================================================
# KELAS ENVIRONMENT E-PUCK (STANDAR GYMNASIUM)
# ==============================================================================
class EPuckGymEnv(gym.Env):
    """
    Environment Gymnasium untuk robot e-puck di Webots.
    Mengintegrasikan sensor, aktuator, dan logika reward/penalty.
    """
    metadata = {'render_modes': []}

    def __init__(self, robot: Supervisor, config: Dict[str, Any]):
        super().__init__()
        self.robot = robot
        self.config = config["env_params"]
        self.timestep: int = int(self.robot.getBasicTimeStep())

        # Inisialisasi Sensor dan Motor
        self._setup_devices()

        # Atribut Environment
        self.max_lives: int = self.config["max_lives"]
        self.robot_node = self.robot.getSelf()
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        # Inisialisasi State
        self.ground_buffers: Dict[str, List[float]] = {name: [] for name in self.ground_sensor_names}
        self.buffer_size: int = self.config["buffer_size"]
        
        # Definisi Ruang Aksi dan Observasi
        state_size = len(self.ground_sensor_names) + len(self.prox_sensor_names) + self.config["num_lidar_sectors"]
        self.action_space = spaces.Discrete(len(self.config["action_space_velocities"]))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)

        # Kalibrasi Sensor (asumsi nilai min/max)
        self.ground_calibration: Dict[str, Tuple[float, float]] = {
            name: (9.0, 1000.0) for name in self.ground_sensor_names
        }
        self.max_proximity_value: float = 1024.0

        # Manajemen Teleportasi
        self._teleported_recently: bool = False
        self._teleport_cooldown: int = 0 # Jumlah langkah untuk mengabaikan tabrakan setelah teleport

    def _setup_devices(self):
        """Menginisialisasi semua perangkat keras (sensor dan motor) pada robot."""
        self.ground_sensor_names: List[str] = ['irR', 'irL', 'irGR', 'irGL', 'irCL', 'irCR']
        self.ground_sensors = {name: self.robot.getDevice(name) for name in self.ground_sensor_names}
        for sensor in self.ground_sensors.values(): sensor.enable(self.timestep)

        self.prox_sensor_names: List[str] = [f'ps{i}' for i in range(8)]
        self.prox_sensors = {name: self.robot.getDevice(name) for name in self.prox_sensor_names}
        for sensor in self.prox_sensors.values(): sensor.enable(self.timestep)

        self.lidar: Lidar = self.robot.getDevice('e_puck_lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            self.lidar_max_range: float = self.lidar.getMaxRange()
            self.lidar_num_beams: int = self.lidar.getHorizontalResolution()
        else:
            self.lidar_max_range, self.lidar_num_beams = 1.0, 0

        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)

    def _get_state(self) -> np.ndarray:
        """Membaca semua sensor dan menyusun vektor state yang dinormalisasi."""
        state_numeric: List[float] = []

        # Ground Sensors
        for name, sensor in self.ground_sensors.items():
            self.ground_buffers[name].append(sensor.getValue())
            if len(self.ground_buffers[name]) > self.buffer_size:
                self.ground_buffers[name].pop(0)
            state_numeric.append(self._normalize_ground_sensor(np.mean(self.ground_buffers[name]), name))

        # Proximity Sensors
        for sensor in self.prox_sensors.values():
            val = max(0.0, min(1.0, sensor.getValue() / self.max_proximity_value))
            state_numeric.append(val)

        # Lidar Sectors
        if self.lidar and self.lidar_num_beams > 0:
            point_cloud = self.lidar.getPointCloud()
            if point_cloud and len(point_cloud) > 0:
                ranges = [np.sqrt(p.x**2 + p.y**2) for p in point_cloud]
                sector_size = self.lidar_num_beams // self.config["num_lidar_sectors"]
                lidar_sector_values = []
                for i in range(self.config["num_lidar_sectors"]):
                    sector_ranges = ranges[i * sector_size:(i + 1) * sector_size]
                    valid_ranges = [r for r in sector_ranges if np.isfinite(r)]
                    min_dist = min(valid_ranges) if valid_ranges else self.lidar_max_range
                    normalized_dist = min(min_dist, self.lidar_max_range) / self.lidar_max_range
                    lidar_sector_values.append(normalized_dist)
                state_numeric.extend(lidar_sector_values)
            else:
                state_numeric.extend([1.0] * self.config["num_lidar_sectors"])
        
        return np.array(state_numeric, dtype=np.float32)

    def _normalize_ground_sensor(self, raw_value: float, sensor_name: str) -> float:
        """Normalisasi nilai sensor tanah ke rentang [0, 1]."""
        min_val, max_val = self.ground_calibration[sensor_name]
        if (max_val - min_val) == 0: return 0.0
        # Nilai dibalik karena sensor IR Webots memberi nilai lebih tinggi untuk permukaan gelap (garis)
        return max(0.0, min(1.0, (max_val - raw_value) / (max_val - min_val)))

    def _reset_simulation_state(self, track_id: int):
        """Mereset posisi dan fisika robot ke checkpoint yang ditentukan."""
        cp = self.config["checkpoints"].get(track_id, [0, 0, 0])
        self.translation_field.setSFVec3f(cp)
        self.rotation_field.setSFRotation([0, 0, 1, 0])
        self.robot_node.resetPhysics()
        # Penting: Berikan beberapa langkah simulasi agar fisika stabil setelah reset
        for _ in range(5): self.robot.step(self.timestep)
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        for name in self.ground_sensor_names: self.ground_buffers[name].clear()
        # Reset flag teleportasi dan cooldown saat status simulasi direset
        self._teleported_recently = False
        self._teleport_cooldown = 0

        # Tambahkan dorongan maju kecil setelah teleportasi untuk membantu robot menjauh dari titik tabrakan
        initial_forward_steps = 5 # Gerakan maju selama 5 langkah simulasi
        initial_speed = 1.0 # Kecepatan maju kecil
        self.left_motor.setVelocity(initial_speed)
        self.right_motor.setVelocity(initial_speed)
        for _ in range(initial_forward_steps):
            self.robot.step(self.timestep)
        self.left_motor.setVelocity(0.0) # Hentikan motor setelah dorongan
        self.right_motor.setVelocity(0.0)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Mereset environment untuk episode baru."""
        super().reset(seed=seed)
        self.lives = self.max_lives
        self._reset_simulation_state(options.get("track_id", 0) if options else 0)

        # Reset statistik episode
        self.episode_reward_breakdown = {'track': 0.0, 'action': 0.0, 'obstacle': 0.0, 'avoid': 0.0}
        self.episode_collision_events = 0 # Reset total collision events for new episode

        # Pastikan flag teleportasi direset untuk episode baru
        self._teleported_recently = False
        self._teleport_cooldown = 0

        return self._get_state(), {}

    def teleport_to_checkpoint(self, track_id: int):
        """Teleportasi robot ke checkpoint tanpa mereset seluruh episode."""
        logging.info(f"Robot diteleportasi ke checkpoint {track_id}.") # Log teleportasi
        self._reset_simulation_state(track_id)
        # Saat teleportasi, aktifkan cooldown
        self._teleported_recently = True
        self._teleport_cooldown = 30 # Jumlah langkah untuk mengabaikan tabrakan setelah teleport (ditingkatkan)
        return self._get_state()

    def _calculate_rewards(self, action: int) -> Tuple[float, str, bool]:
        """Menghitung total reward dan alasan terminasi jika ada."""
        total_reward = 0.0
        termination_reason = ""
        
        # Dapatkan nilai sensor jarak saat ini
        prox_values = [s.getValue() for s in self.prox_sensors.values()]
        
        # Deteksi tabrakan berdasarkan nilai sensor mentah
        raw_is_colliding = any(v > self.config["proximity_collision_threshold"] * self.max_proximity_value for v in prox_values)
        collided_in_this_step = raw_is_colliding # Flag ini mencerminkan status sensor aktual

        # Deteksi penghindaran berdasarkan nilai sensor mentah
        is_avoiding = any(v > self.config["proximity_avoid_threshold"] * self.max_proximity_value for v in prox_values)

        # Tentukan apakah tabrakan harus dipertimbangkan untuk reward/penalti (mengabaikan jika baru diteleportasi)
        is_colliding_for_rewards = raw_is_colliding and not self._teleported_recently

        # 1. Reward Aksi & Penalti Waktu
        action_reward = self.config["reward_action_straight"] if action == 0 else self.config["penalty_action_turn"]
        total_reward += action_reward + self.config["penalty_time_step"]
        self.episode_reward_breakdown['action'] += action_reward

        # 2. Reward Menghindari & Penalti Tabrakan
        if is_colliding_for_rewards: # Gunakan ini untuk nyawa dan penalti
            obstacle_reward = self.config["penalty_collision"]
            self.episode_collision_events += 1 # Ini adalah penghitung kumulatif untuk episode
            self.lives -= 1 # Nyawa berkurang saat terjadi tabrakan
            if self.lives <= 0:
                termination_reason = TERMINATION_REASON_NO_LIVES
        elif is_avoiding and not is_colliding_for_rewards: # Beri reward menghindari jika tidak dalam kondisi tabrakan yang dihukum
            obstacle_reward = self.config["reward_avoid_collision"]
        else:
            obstacle_reward = 0.0
            
        total_reward += obstacle_reward
        self.episode_reward_breakdown['obstacle'] += self.config["penalty_collision"] if is_colliding_for_rewards else 0
        self.episode_reward_breakdown['avoid'] += self.config["reward_avoid_collision"] if is_avoiding and not is_colliding_for_rewards else 0

        # 3. Reward Mengikuti Jalur (hanya jika belum ada terminasi lain dari tabrakan fatal)
        if not termination_reason: # Jika belum ada terminasi karena kehabisan nyawa
            ground_vals = [self._normalize_ground_sensor(s.getValue(), n) for n, s in self.ground_sensors.items()]
            on_center = (ground_vals[4] > self.config["center_line_detect_threshold"] and
                         ground_vals[5] > self.config["center_line_detect_threshold"])
            on_track = any(s > self.config["line_detect_threshold"] for s in ground_vals)

            if on_center:
                track_reward = self.config["reward_on_track_center"]
            elif on_track:
                track_reward = self.config["reward_on_track_edge"]
            else:
                track_reward = self.config["penalty_off_track"]
                termination_reason = TERMINATION_REASON_OFF_TRACK # Robot keluar jalur
                
            total_reward += track_reward
            self.episode_reward_breakdown['track'] += track_reward

        return total_reward, termination_reason, collided_in_this_step # Mengembalikan flag baru

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Menjalankan satu langkah di environment."""
        if self.robot.step(self.timestep) == -1:
            # Jika simulasi berhenti, akhiri episode
            return self._get_state(), 0.0, True, True, {}

        # Kurangi cooldown teleportasi
        if self._teleport_cooldown > 0:
            self._teleport_cooldown -= 1
            if self._teleport_cooldown == 0:
                self._teleported_recently = False # Nonaktifkan flag setelah cooldown selesai

        # Terapkan aksi ke motor
        left_speed, right_speed = self.config["action_space_velocities"][action]
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Hitung reward, tentukan alasan terminasi, dan dapatkan status tabrakan dari _calculate_rewards
        # `collided_in_this_step` akan bernilai True hanya jika tabrakan terdeteksi di langkah ini
        reward, termination_reason_from_calc, collided_in_this_step = self._calculate_rewards(action)
        
        terminated = False
        truncated = False
        info = {}

        # Tentukan apakah episode berakhir
        if self.lives <= 0:
            terminated = True
            info['termination_reason'] = TERMINATION_REASON_NO_LIVES
        elif termination_reason_from_calc == TERMINATION_REASON_OFF_TRACK:
            terminated = True
            info['termination_reason'] = TERMINATION_REASON_OFF_TRACK
        
        # Logika teleportasi:
        # Jika terjadi tabrakan *di langkah ini* (collided_in_this_step adalah True)
        # DAN episode belum berakhir (artinya masih ada nyawa tersisa setelah tabrakan ini)
        # DAN robot tidak baru saja diteleportasi (tidak dalam masa cooldown)
        # maka teleportasi robot ke checkpoint.
        # Ini memastikan robot hanya diteleportasi sekali per insiden tabrakan non-fatal.
        if collided_in_this_step and not terminated and not self._teleported_recently:
            self.teleport_to_checkpoint(self.config.get("track_id", 0))
            # Flag `_teleported_recently` dan `_teleport_cooldown` akan diatur di `teleport_to_checkpoint`

        # Ambil observasi berikutnya *setelah* potensi teleportasi
        next_observation = self._get_state()

        if terminated:
            info['reward_breakdown'] = self.episode_reward_breakdown
            info['lives_lost'] = self.max_lives - self.lives
            info['collision_events'] = self.episode_collision_events

        return next_observation, reward, terminated, truncated, info

# ==============================================================================
# FUNGSI UTAMA PELATIHAN
# ==============================================================================
def main(config: Dict[str, Any]):
    """Fungsi utama untuk menjalankan seluruh proses pelatihan."""
    setup_logging()
    robot = Supervisor()
    
    # Stabilisasi awal dunia Webots
    for _ in range(10):
        robot.step(int(robot.getBasicTimeStep()))

    # Persiapan path dan direktori
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, config["models_dir"])
    os.makedirs(models_dir, exist_ok=True)
    
    # --- PERBAIKAN: Memisahkan path untuk load dan save model ---
    # Path untuk model yang akan DIMUAT (model dari fase sebelumnya)
    # Anda perlu mengganti nama file ini sesuai dengan model yang ingin Anda lanjutkan
    load_model_filename = "dqn_model_Fase1.03.zip" # <-- GANTI INI sesuai model yang ingin dimuat
    load_model_path = os.path.join(models_dir, load_model_filename)

    # Path untuk model yang akan DISIMPAN (model hasil pelatihan ini)
    # Ini akan mengambil nama dari config["model_filename"]
    save_model_path = os.path.join(models_dir, config["model_filename"])
    
    logging.info(f"Dunia aktif: {os.path.basename(robot.getWorldPath())}")
    logging.info(f"Memulai siklus: {config['training_cycle']} - {config['phase_name']}")

    # Inisialisasi Environment dan Model
    env = EPuckGymEnv(robot=robot, config=config)
    env = Monitor(env)
    check_env(env)

    # Logika memuat atau membuat model
    if os.path.exists(load_model_path): # <-- Cek keberadaan model yang akan dimuat
        logging.info(f"Memuat model yang ada dari: {load_model_path}")
        model = DQN.load(load_model_path, env=env)
        
        # Setel ulang learning rate jika perlu
        model.learning_rate = config["dqn_params"]["learning_rate"]
        
        # --- PERBAIKAN: Mengatur ulang Epsilon untuk melanjutkan pelatihan ---
        # Mengatur ulang exploration_rate ke nilai awal yang baru
        model.exploration_rate = config["dqn_params"]["exploration_initial_eps"]
        # MENGATUR ULANG NUM_TIMESTEPS MODEL UNTUK MEMULAI JADWAL EXPLORASI DARI AWAL
        model.num_timesteps = 0 
        
    else:
        logging.info(f"Model '{load_model_filename}' tidak ditemukan. Membuat model DQN baru.")
        tensorboard_log_path = os.path.join(models_dir, config["tensorboard_log_dir"])
        model = DQN(
            env=env,
            tensorboard_log=tensorboard_log_path,
            **config["dqn_params"]
        )

    # Pelatihan
    total_timesteps = config["training"]["num_episodes"] * config["training"]["max_steps_per_episode"]
    custom_callback = CustomLoggerCallback(config=config)
    
    logging.info(f"===== MEMULAI PELATIHAN: {config['phase_name']} ({total_timesteps} total langkah) =====")
    phase_start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False, # Ini penting agar total timesteps callback terus bertambah
            callback=custom_callback
        )
    except Exception as e:
        logging.error(f"Terjadi error saat pelatihan '{config['phase_name']}': {e}", exc_info=True)
        # Simpan model darurat
        emergency_path = save_model_path.replace(".zip", "_emergency.zip") # Menggunakan save_model_path
        model.save(emergency_path)
        logging.info(f"Model darurat disimpan di: {emergency_path}")
        raise

    model.save(save_model_path) # <-- Menggunakan save_model_path untuk menyimpan
    logging.info(f"Pelatihan '{config['phase_name']}' selesai. Model disimpan di: {save_model_path}")

    # Ringkasan Fase
    phase_duration = time.time() - phase_start_time
    summary = custom_callback.summary
    num_episodes = custom_callback.episode_count
    avg_lives_lost = summary["Total Lives Lost"] / num_episodes if num_episodes > 0 else 0

    logging.info(f"\n===== RINGKASAN PELATIHAN: {config['phase_name']} =====")
    logging.info(f"Durasi Training: {format_duration(phase_duration)}")
    logging.info(f"Total Step Dijalani: {summary['Total Steps']}")
    logging.info(f"Total Episode Selesai (Agen Selalu di Jalur): {summary[TERMINATION_REASON_MAX_STEPS]}")
    logging.info(f"Total Episode Gagal (Off Track): {summary[TERMINATION_REASON_OFF_TRACK]}")
    logging.info(f"Total Episode Gagal (Tabrakan Fatal): {summary[TERMINATION_REASON_NO_LIVES]}")
    logging.info(f"Total Insiden Tabrakan (Non-Fatal): {summary['Collision Events']}")
    logging.info(f"Rata-rata Kehilangan Nyawa per Episode: {avg_lives_lost:.2f}")
    logging.info("=" * 50 + "\n")


if __name__ == "__main__":
    main(TRAINING_CONFIG)
