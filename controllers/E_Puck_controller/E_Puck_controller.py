import os
import time
import logging
import random
from typing import List, Dict, Tuple, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import dari Stable-Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# Import dari Webots
try:
    from controller import Supervisor, Lidar, Node
except ImportError:
    print("Peringatan: Modul 'controller' Webots tidak ditemukan. Skrip ini hanya akan berjalan di luar Webots untuk pengujian.")
    # Definisikan kelas placeholder untuk pengujian di luar Webots
    class Device:
        def enable(self, timestep): pass
        def getValue(self): return 0.0
    class Motor(Device):
        def setPosition(self, pos): pass
        def setVelocity(self, vel): pass
    class Lidar(Device):
        def enablePointCloud(self): pass
        def getMaxRange(self): return 1.0
        def getHorizontalResolution(self): return 0
        def getPointCloud(self): return []
    class Node:
        def getPosition(self): return [0.0, 0.0, 0.0]
        def getField(self, field_name): return type('Field', (object,), {'setSFVec3f': lambda s, v: None, 'setSFRotation': lambda s, r: None})()
        def resetPhysics(self): pass
    class DummyRobot:
        def getBasicTimeStep(self): return 32
        def getSelf(self) -> Node: return Node()
        def step(self, timestep): return 0
        def getWorldPath(self): return "dummy_world.wbt"
        def getDevice(self, name):
            if 'motor' in name: return Motor()
            if 'lidar' in name: return Lidar()
            return Device()
    Supervisor = DummyRobot

# ==============================================================================
# KONFIGURASI UTAMA (Tidak diubah)
# ==============================================================================
TRAINING_CONFIG = {
    "phase_name": "Training Fase 2 - 1",
    "training_cycle": 1,
    "track_id": 1, 
    "models_dir": "saved_models_sb3",
    "load_model_filename": None,
    "model_filename": "dqn_fase4_model.zip",
    "tensorboard_log_dir": "tensorboard_logs/",
    
    "training": {
        "num_episodes": 1500,
        "max_steps_per_episode": 1000,
        "algorithm": "DQN",
    },
    
    "dqn_params": {
        "policy": "MlpPolicy",
        "learning_rate": 0.00025,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": (4, "step"),
        "target_update_interval": 1000,
        "verbose": 0,
        "exploration_fraction": 0.5, # Sesuai diskusi kita, dinaikkan
        "exploration_final_eps": 0.05,
        "policy_kwargs": {
            "net_arch": [128, 128]
        }
    },

    "env_params": {
        "max_lives": 3,
        "buffer_size": 5,
        "num_lidar_sectors": 8,
        "action_space_velocities": [(6.28, 6.28), (3.14, 6.28), (6.28, 3.14)],
        "checkpoints": {
            0: [-1.62948, 1.01949, -0.00963313],
            1: [-1.89295, 1.61571, -0.00988564],
            2: [0.0, 0.0, 0.0]
        },
        
        "line_detect_threshold": 0.4,
        "center_line_detect_threshold": 0.8,
        "proximity_collision_threshold": 0.7,

        "reward_on_track_center": 20.0,
        "reward_on_track_edge": 2.0,
        "penalty_time_step": -0.05,
        "penalty_collision": -30.0,
        "penalty_distance_from_line_factor": -2.0,
        
        # Parameter baru untuk detektor macet
        "stuck_detector_threshold": 0.001, # Jarak minimal pergerakan
        "stuck_detector_patience": 50,     # Jumlah langkah untuk dianggap macet
        
        "shaping_weights": {
            0: {"track": 10.0, "obstacle": 0.0, "speed": 0.0},
            1: {"track": 10.0, "obstacle": 5.0, "speed": 0.0},
            2: {"track": 8.0, "obstacle": 6.0, "speed": 2.0}
        },

        "sensor_calibration": {
            "ground_sensors": {
                "irR": [200.0, 950.0], "irL": [200.0, 950.0], "irGR": [300.0, 1000.0],
                "irGL": [300.0, 1000.0], "irCL": [400.0, 1000.0], "irCR": [400.0, 1000.0]
            },
            "prox_sensors": {
                "ps0": [80.0, 1024.0], "ps7": [80.0, 1024.0], "ps1": [80.0, 800.0],
                "ps6": [80.0, 800.0], "ps2": [80.0, 600.0], "ps5": [80.0, 600.0],
                "ps3": [80.0, 500.0], "ps4": [80.0, 500.0]
            },
            "lidar": {"min_range": 0.0, "max_range": 0.5}
        }
    }
}

# ==============================================================================
# KONSTANTA (DENGAN TAMBAHAN)
# ==============================================================================
TERMINATION_REASON_MAX_STEPS = "Max Steps"
TERMINATION_REASON_OFF_TRACK = "Off Track"
TERMINATION_REASON_NO_LIVES = "No Lives Left"
TERMINATION_REASON_STUCK = "Stuck" # Alasan terminasi baru

# ==============================================================================
# FUNGSI UTILITAS (Tidak diubah)
# ==============================================================================
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, minutes, secs = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{hours} jam, {minutes} menit, {secs} detik"

# ==============================================================================
# KELAS NORMALISASI SENSOR (Tidak diubah)
# ==============================================================================
class SensorNormalization:
    @staticmethod
    def min_max_scaling(raw_value: float, min_val: float, max_val: float, inverted=False) -> float:
        if max_val - min_val == 0: return 0.0
        normalized = (raw_value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        return 1.0 - normalized if inverted else normalized

# ==============================================================================
# KELAS CALLBACK (DENGAN PENYESUAIAN)
# ==============================================================================
class CustomLoggerCallback(BaseCallback):
    def __init__(self, config: Dict[str, Any], verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.phase_name = config["phase_name"]
        self.training_cycle = config["training_cycle"]
        self.num_episodes = config["training"]["num_episodes"]
        self.episode_count = 0
        self.summary = {
            TERMINATION_REASON_MAX_STEPS: 0, TERMINATION_REASON_OFF_TRACK: 0,
            TERMINATION_REASON_NO_LIVES: 0, TERMINATION_REASON_STUCK: 0, # Ditambahkan
            "Collision Events": 0, "Total Lives Lost": 0, "Total Steps": 0
        }

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1
            info = self.locals['infos'][0]
            if 'episode' in info:
                ep_rew, ep_len = info['episode']['r'], info['episode']['l']
                term_reason = info.get('termination_reason') or TERMINATION_REASON_MAX_STEPS
                self.summary[term_reason] = self.summary.get(term_reason, 0) + 1
                self.summary["Total Lives Lost"] += info.get('lives_lost', 0)
                self.summary["Collision Events"] += info.get('collision_events', 0)
                epsilon = self.model.exploration_rate if hasattr(self.model, 'exploration_rate') else -1
                
                log_msg = (
                    f"Siklus {self.training_cycle} | {self.phase_name} | Ep {self.episode_count}/{self.num_episodes} | "
                    f"Reward: {ep_rew:.2f} | Steps: {ep_len} | Epsilon: {epsilon:.4f} | "
                    f"Terminasi: {term_reason}"
                )
                logging.info(log_msg)
        return True

    def _on_training_end(self) -> None:
        self.summary["Total Steps"] = self.num_timesteps

# ==============================================================================
# KELAS ENVIRONMENT E-PUCK (DENGAN DETEKTOR MACET)
# ==============================================================================
class EPuckGymEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, robot: Supervisor, config: Dict[str, Any]):
        super().__init__()
        self.robot = robot
        self.config = config["env_params"]
        self.dqn_config = config["dqn_params"]
        self.track_id = config["track_id"]
        
        self.timestep: int = int(self.robot.getBasicTimeStep())
        self._setup_devices()
        
        self.max_lives: int = self.config["max_lives"]
        self.lives: int = self.max_lives
        
        self.robot_node: Node = self.robot.getSelf()
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        state_size = len(self.ground_sensor_names) + len(self.prox_sensor_names) + self.config["num_lidar_sectors"]
        self.action_space = spaces.Discrete(len(self.config["action_space_velocities"]))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)
        
        self.previous_potential: float = 0.0
        self.episode_collision_events: int = 0
        self.steps_off_track: int = 0
        
        # Variabel baru untuk detektor macet
        self.stuck_counter: int = 0
        self.last_position: Optional[List[float]] = None

    def _setup_devices(self):
        self.ground_sensor_names = ['irR', 'irL', 'irGR', 'irGL', 'irCL', 'irCR']
        self.prox_sensor_names = [f'ps{i}' for i in range(8)]
        
        self.ground_sensors = {name: self.robot.getDevice(name) for name in self.ground_sensor_names}
        self.prox_sensors = {name: self.robot.getDevice(name) for name in self.prox_sensor_names}
        for sensor in list(self.ground_sensors.values()) + list(self.prox_sensors.values()):
            sensor.enable(self.timestep)
        
        self.lidar: Lidar = self.robot.getDevice('e_puck_lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.lidar_num_beams: int = self.lidar.getHorizontalResolution()
        
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

    def _get_state(self) -> np.ndarray:
        state_numeric: List[float] = []
        
        calib = self.config["sensor_calibration"]["ground_sensors"]
        for name in self.ground_sensor_names:
            min_val, max_val = calib[name]
            state_numeric.append(SensorNormalization.min_max_scaling(self.ground_sensors[name].getValue(), min_val, max_val, inverted=True))
        
        calib = self.config["sensor_calibration"]["prox_sensors"]
        for name in self.prox_sensor_names:
            min_val, max_val = calib[name]
            state_numeric.append(SensorNormalization.min_max_scaling(self.prox_sensors[name].getValue(), min_val, max_val))
        
        max_range = self.config["sensor_calibration"]["lidar"]["max_range"]
        point_cloud = self.lidar.getPointCloud()
        if point_cloud and self.lidar_num_beams > 0:
            ranges = [np.sqrt(p.x**2 + p.y**2) for p in point_cloud]
            sector_size = self.lidar_num_beams // self.config["num_lidar_sectors"]
            for i in range(self.config["num_lidar_sectors"]):
                sector_ranges = ranges[i * sector_size:(i + 1) * sector_size]
                min_dist = min((r for r in sector_ranges if np.isfinite(r)), default=max_range)
                state_numeric.append(SensorNormalization.min_max_scaling(min_dist, 0.0, max_range, inverted=True))
        else:
            state_numeric.extend([0.0] * self.config["num_lidar_sectors"])
            
        return np.array(state_numeric, dtype=np.float32)
    
    def _calculate_potential(self, state: np.ndarray, action: int) -> float:
        weights = self.config["shaping_weights"][self.track_id]
        phi_track = state[4] + state[5]
        prox_values = state[6:14]
        phi_obstacle = np.sum(1.0 - prox_values)
        phi_speed = 1.0 if action == 0 else 0.5
        return weights["track"] * phi_track + weights["obstacle"] * phi_obstacle + weights["speed"] * phi_speed

    def _calculate_base_rewards(self, state: np.ndarray) -> Tuple[float, str, bool]:
        prox_vals, ground_vals = state[6:14], state[0:6]
        is_colliding = any(v > self.config["proximity_collision_threshold"] for v in prox_vals)
        termination_reason = ""
        base_reward = self.config["penalty_time_step"]
        
        if is_colliding:
            base_reward += self.config["penalty_collision"]
            self.episode_collision_events += 1
            self.lives -= 1
            if self.lives <= 0:
                termination_reason = TERMINATION_REASON_NO_LIVES
        
        if not termination_reason:
            is_on_center = any(g > self.config["center_line_detect_threshold"] for g in ground_vals[4:6])
            is_on_edge = any(g > self.config["line_detect_threshold"] for g in ground_vals)

            if is_on_center:
                base_reward += self.config["reward_on_track_center"]
                self.steps_off_track = 0
            elif is_on_edge:
                base_reward += self.config["reward_on_track_edge"]
                self.steps_off_track = 0
            else:
                self.steps_off_track += 1
                distance_penalty = self.steps_off_track * self.config["penalty_distance_from_line_factor"]
                base_reward += distance_penalty
                
                if self.steps_off_track > 100:
                    termination_reason = TERMINATION_REASON_OFF_TRACK

        return base_reward, termination_reason, is_colliding
    
    def _check_if_stuck(self) -> bool:
        """Memeriksa apakah robot macet dengan membandingkan posisi saat ini dan sebelumnya."""
        current_position = self.robot_node.getPosition()
        if self.last_position is not None:
            distance_moved = np.linalg.norm(np.array(current_position) - np.array(self.last_position))
            if distance_moved < self.config["stuck_detector_threshold"]:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0 # Reset jika bergerak
        
        self.last_position = current_position
        return self.stuck_counter >= self.config["stuck_detector_patience"]

    def _perform_evasive_maneuver(self):
        logging.info("Tabrakan terdeteksi! Melakukan manuver menghindar yang lebih terkontrol.")
        self.left_motor.setVelocity(-2.0)
        self.right_motor.setVelocity(-2.0)
        for _ in range(15):
            if self.robot.step(self.timestep) == -1: return

        turn_speed = 2.5
        turn_direction = 1 if random.random() > 0.5 else -1
        self.left_motor.setVelocity(turn_speed * turn_direction)
        self.right_motor.setVelocity(-turn_speed * turn_direction)
        for _ in range(20):
            if self.robot.step(self.timestep) == -1: return

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.robot.step(self.timestep)

    def _reset_simulation_state(self):
        cp = self.config["checkpoints"][self.track_id]
        self.translation_field.setSFVec3f(cp)
        self.rotation_field.setSFRotation([0, 0, 1, random.uniform(-0.26, 0.26)])
        self.robot_node.resetPhysics()
        
        for _ in range(5):
            self.robot.step(self.timestep)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_simulation_state()
        
        self.lives = self.max_lives
        self.episode_collision_events = 0
        self.steps_off_track = 0
        
        # Reset detektor macet
        self.stuck_counter = 0
        self.last_position = self.robot_node.getPosition()
        
        initial_state = self._get_state()
        self.previous_potential = self._calculate_potential(initial_state, 0)
        
        return initial_state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.robot.step(self.timestep) == -1:
            return self._get_state(), 0.0, True, False, {}

        # Terapkan aksi
        left_speed, right_speed = self.config["action_space_velocities"][action]
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        # Ambil state dan hitung reward
        current_state = self._get_state()
        base_reward, termination_reason, collided = self._calculate_base_rewards(current_state)
        
        # Periksa kondisi macet
        if not termination_reason and self._check_if_stuck():
            termination_reason = TERMINATION_REASON_STUCK
            base_reward -= 10 # Beri penalti tambahan karena macet
            
        terminated = bool(termination_reason)
        next_state = current_state

        # Lakukan manuver jika terjadi tabrakan non-fatal
        if collided and not terminated:
            self._perform_evasive_maneuver()
            next_state = self._get_state()
            self.previous_potential = self._calculate_potential(next_state, action)
            # Reset posisi terakhir setelah manuver untuk menghindari deteksi macet yang salah
            self.last_position = self.robot_node.getPosition()
        
        # Hitung reward shaping
        current_potential = self._calculate_potential(next_state, action)
        shaping_reward = (self.dqn_config["gamma"] * current_potential) - self.previous_potential
        self.previous_potential = current_potential
        total_reward = base_reward + shaping_reward

        info = {
            'lives_lost': self.max_lives - self.lives,
            'collision_events': self.episode_collision_events,
            'termination_reason': termination_reason if terminated else None
        }
        
        return next_state, float(total_reward), terminated, False, info

# ==============================================================================
# FUNGSI UTAMA PELATIHAN (Tidak diubah)
# ==============================================================================
def main(config: Dict[str, Any]):
    setup_logging()
    robot = Supervisor()
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd() 
        
    models_dir = os.path.join(base_dir, config["models_dir"])
    os.makedirs(models_dir, exist_ok=True)
    
    env = EPuckGymEnv(robot=robot, config=config)
    env = Monitor(env)
    check_env(env)

    load_model_path = config.get("load_model_filename")
    if load_model_path:
        load_model_path = os.path.join(models_dir, load_model_path)

    if load_model_path and os.path.exists(load_model_path):
        logging.info(f"Memuat model yang ada dari: {load_model_path}")
        model = DQN.load(load_model_path, env=env)
        model.learning_rate = config["dqn_params"]["learning_rate"]
    else:
        logging.info("Membuat model DQN baru.")
        tensorboard_log_path = os.path.join(base_dir, config["tensorboard_log_dir"])
        model = DQN(
            env=env,
            tensorboard_log=tensorboard_log_path,
            **config["dqn_params"]
        )

    total_timesteps = config["training"]["num_episodes"] * config["training"]["max_steps_per_episode"]
    custom_callback = CustomLoggerCallback(config=config)
    save_model_path = os.path.join(models_dir, config["model_filename"])
    
    logging.info(f"===== MEMULAI PELATIHAN: {config['phase_name']} ({total_timesteps} total langkah) =====")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=not (load_model_path and os.path.exists(load_model_path)),
            callback=custom_callback
        )
        model.save(save_model_path)
        logging.info(f"Pelatihan selesai. Model disimpan di: {save_model_path}")
    except Exception as e:
        logging.error(f"Terjadi error saat pelatihan: {e}", exc_info=True)
        emergency_path = save_model_path.replace(".zip", "_emergency.zip")
        model.save(emergency_path)
        logging.info(f"Model darurat disimpan di: {emergency_path}")
        raise

    duration = time.time() - start_time
    summary = custom_callback.summary
    num_episodes = custom_callback.episode_count
    avg_lives_lost = summary["Total Lives Lost"] / num_episodes if num_episodes > 0 else 0

    logging.info(f"\n===== RINGKASAN PELATIHAN: {config['phase_name']} =====")
    logging.info(f"Durasi Training: {format_duration(duration)}")
    logging.info(f"Total Step Dijalani: {summary['Total Steps']}")
    logging.info(f"Total On-Track: {summary[TERMINATION_REASON_MAX_STEPS]}")
    logging.info(f"Total Episode Gagal (Off Track): {summary[TERMINATION_REASON_OFF_TRACK]}")
    logging.info(f"Total Episode Gagal (Tabrakan Fatal): {summary[TERMINATION_REASON_NO_LIVES]}")
    logging.info(f"Total Episode Gagal (Macet): {summary[TERMINATION_REASON_STUCK]}") # Ditambahkan
    logging.info(f"Total Insiden Tabrakan (Non-Fatal): {summary['Collision Events']}")
    logging.info(f"Rata-rata Kehilangan Nyawa per Episode: {avg_lives_lost:.2f}")
    logging.info("=" * 50 + "\n")


if __name__ == "__main__":
    main(TRAINING_CONFIG)
