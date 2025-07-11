import numpy as np
import sys
import os
from controller import Supervisor, Lidar, Camera
import logging
import time #Impor modul 'time'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Penyesuaian sys.path
current_controller_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_controller_dir, "..", ".."))
sys.path.append(project_root)
controllers_dir = os.path.join(project_root, "controllers")
if controllers_dir not in sys.path:
    sys.path.append(controllers_dir)

# Impor modul agen dan pelatihan
from dqn_agent.dqn_agent import DQNAgent 
from Training_controller.Training_controller import train_agent, evaluate_agent 

logging.info("Core modules (DQNAgent, Training_controller) imported successfully.")

class EPuckEnv:
    def __init__(self, robot: Supervisor, track_id: int = 0):
        self.robot = robot
        self.timestep: int = int(self.robot.getBasicTimeStep())

        # Inisialisasi Sensor
        self.ground_sensor_names: list[str] = ['irR', 'irL', 'irGR', 'irGL', 'irCL', 'irCR'] 
        self.ground_sensors = {name: self.robot.getDevice(name) for name in self.ground_sensor_names}
        for sensor in self.ground_sensors.values(): sensor.enable(self.timestep)

        self.prox_sensor_names: list[str] = [f'ps{i}' for i in range(8)]
        self.prox_sensors = {name: self.robot.getDevice(name) for name in self.prox_sensor_names}
        for sensor in self.prox_sensors.values(): sensor.enable(self.timestep)
            
        self.lidar: Lidar = self.robot.getDevice('e_puck_lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud() # Penting untuk getPointCloud()
            logging.info("LiDAR sensor initialized successfully.")
            self.lidar_max_range: float = self.lidar.getMaxRange()
            self.lidar_num_beams: int = self.lidar.getHorizontalResolution()
            self.num_lidar_sectors: int = 8
        else:
            logging.warning("LiDAR sensor 'e_puck_lidar' not found.")
            self.lidar_max_range, self.lidar_num_beams, self.num_lidar_sectors = 1.0, 360, 0

        # Inisialisasi Kamera (untuk visualisasi)
        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)
            logging.info("Camera enabled for visual overlay.")
        else:
            logging.warning("Camera device 'camera' not found.")

        # Inisialisasi Motor
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)

        self.max_lives: int = 3
        self.lives: int = self.max_lives

        # Node robot
        self.robot_node = self.robot.getSelf()
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.track_id: int = track_id
        self.checkpoints: dict[int, list[float]] = {0: [-1.90728, 1.60962, -0.00243969]}

        # Kalibrasi
        self.ground_calibration: dict[str, tuple[float, float]] = {name: (9.0, 1000.0) for name in self.ground_sensor_names}
        self.LINE_DETECT_THRESHOLD_NORMALIZED: float = 0.5 
        self.CENTER_LINE_DETECT_THRESHOLD_NORMALIZED: float = 0.8 
        self.max_proximity_value: float = 4096.0
        self.PROXIMITY_COLLISION_THRESHOLD_NORMALIZED: float = 0.7 

        self.ground_buffers: dict[str, list[float]] = {name: [] for name in self.ground_sensor_names}
        self.buffer_size: int = 5

        self.action_space_velocities: list[tuple[float, float]] = [(6.28, 6.28), (3.14, 6.28), (6.28, 3.14)]
        self.action_size: int = len(self.action_space_velocities)
        self.state_size: int = len(self.ground_sensor_names) + len(self.prox_sensor_names) + self.num_lidar_sectors
        logging.info(f"Calculated State Size: {self.state_size}")

        # Konstanta Reward
        self.REWARD_ON_TRACK_CENTER, self.REWARD_ON_TRACK_EDGE = 2.0, 0.5
        self.PENALTY_TIME_STEP, self.PENALTY_OFF_TRACK, self.PENALTY_COLLISION = -0.01, -3.0, -5.0
        self.REWARD_ACTION_STRAIGHT, self.REWARD_ACTION_TURN = 0.1, -0.05
        
        self.reset()

    def _normalize_ground_sensor(self, raw_value: float, sensor_name: str) -> float:
        min_val, max_val = self.ground_calibration[sensor_name]
        return max(0.0, min(1.0, (max_val - raw_value) / (max_val - min_val))) if (max_val - min_val) != 0 else 0.0

    def get_state(self) -> np.ndarray:
        state: list[float] = []
        for name, sensor in self.ground_sensors.items():
            self.ground_buffers[name].append(sensor.getValue())
            if len(self.ground_buffers[name]) > self.buffer_size: self.ground_buffers[name].pop(0)
            state.append(self._normalize_ground_sensor(np.mean(self.ground_buffers[name]), name))

        for sensor in self.prox_sensors.values():
            state.append(max(0.0, min(1.0, sensor.getValue() / self.max_proximity_value)))

        if self.lidar and self.num_lidar_sectors > 0:
            point_cloud = self.lidar.getPointCloud()
            if point_cloud:
                lidar_ranges = [np.sqrt(p.x**2 + p.y**2 + p.z**2) for p in point_cloud]
                if len(lidar_ranges) == self.lidar_num_beams:
                    sector_size = self.lidar_num_beams // self.num_lidar_sectors
                    lidar_features_normalized = []
                    for i in range(self.num_lidar_sectors):
                        sector_ranges = lidar_ranges[i * sector_size:(i + 1) * sector_size]
                        valid_ranges = [r for r in sector_ranges if np.isfinite(r)]
                        min_dist = min(valid_ranges) if valid_ranges else self.lidar_max_range
                        lidar_features_normalized.append(min(min_dist, self.lidar_max_range) / self.lidar_max_range)
                    state.extend(lidar_features_normalized)
                else:
                    logging.warning(f"LiDAR beam count mismatch. Expected {self.lidar_num_beams}, got {len(lidar_ranges)}.")
                    state.extend([1.0] * self.num_lidar_sectors)
            else:
                logging.warning("LiDAR getPointCloud() returned empty data. Using default values.")
                state.extend([1.0] * self.num_lidar_sectors)
        
        return np.array(state, dtype=np.float32)

    def _reset_simulation(self):
        cp = self.checkpoints.get(self.track_id, [0, 0, 0])
        self.translation_field.setSFVec3f(cp)
        self.rotation_field.setSFRotation([0, 0, 1, 0])
        self.robot_node.resetPhysics()
        for _ in range(5): self.robot.step(self.timestep)
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        for name in self.ground_sensor_names: self.ground_buffers[name].clear()
    
    def reset(self) -> np.ndarray:
        self.lives = self.max_lives
        self._reset_simulation()
        logging.info(f"\n[RESET] New Episode | Lives: {self.lives}")
        return self.get_state()

    def teleport_to_checkpoint(self) -> np.ndarray:
        self._reset_simulation()
        logging.info(f"[TELEPORT] To checkpoint | Lives remaining: {self.lives}")
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        left_speed, right_speed = self.action_space_velocities[action]
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        if self.robot.step(self.timestep) == -1: return np.zeros(self.state_size), 0.0, True, {} 
        
        next_state = self.get_state()
        reward, done, info = 0.0, False, {}
        reward += self.PENALTY_TIME_STEP
        action_reward_component = self.REWARD_ACTION_STRAIGHT if action == 0 else self.REWARD_ACTION_TURN
        reward += action_reward_component

        ground_normalized = next_state[:len(self.ground_sensor_names)]
        on_center = ground_normalized[4] > self.CENTER_LINE_DETECT_THRESHOLD_NORMALIZED and \
                    ground_normalized[5] > self.CENTER_LINE_DETECT_THRESHOLD_NORMALIZED
        on_track = any(s > self.LINE_DETECT_THRESHOLD_NORMALIZED for s in ground_normalized)

        track_reward_component = 0.0
        if on_center: track_reward_component = self.REWARD_ON_TRACK_CENTER
        elif on_track: track_reward_component = self.REWARD_ON_TRACK_EDGE
        else:
            track_reward_component = self.PENALTY_OFF_TRACK
            done, info['reason'] = True, 'Off Track'
        reward += track_reward_component

        prox_start_idx = len(self.ground_sensor_names)
        prox_end_idx = prox_start_idx + len(self.prox_sensor_names)
        proximity_normalized = next_state[prox_start_idx:prox_end_idx]
        
        obstacle_penalty_component = 0.0
        if any(val > self.PROXIMITY_COLLISION_THRESHOLD_NORMALIZED for val in proximity_normalized):
            reward += self.PENALTY_COLLISION 
            obstacle_penalty_component = self.PENALTY_COLLISION
            self.lives -= 1
            logging.info(f"[COLLISION] Detected! Lives left: {self.lives}. Penalty: {self.PENALTY_COLLISION}")
            if self.lives <= 0: done, info['reason'] = True, 'No Lives Left'
            else: self.teleport_to_checkpoint(); info['reason'] = 'Collision'
                
        info['reward_breakdown'] = {'track': track_reward_component, 'action': action_reward_component, 'obstacle': obstacle_penalty_component}
        return next_state, reward, done, info

if __name__ == "__main__":
    robot = Supervisor()
    env = EPuckEnv(robot=robot, track_id=0)
    agent = DQNAgent(env.state_size, env.action_size) 
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # --- BLOK INI TELAH DIPERBARUI ---
    # Membuat nama file unik menggunakan timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"dqn_epuck_{timestamp}.h5"
    model_save_path = os.path.join(models_dir, model_filename)
    # --- AKHIR PEMBARUAN ---
    
    agent.model_save_path = model_save_path 
    logging.info("Robot controller initialized. Starting training...")
    train_agent(agent, env) 
    logging.info("Training process completed.")
    evaluate_agent(agent, env) 
    logging.info("Evaluation process completed.")