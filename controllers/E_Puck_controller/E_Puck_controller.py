import numpy as np
import sys
import os
from controller import Supervisor, Lidar, Camera
import logging
import time

# Mencoba mengimpor OpenCV (cv2) untuk pemrosesan gambar
try:
    import cv2
except ImportError:
    # Memberi pesan error jika OpenCV tidak terinstall, karena fitur kamera tidak akan berfungsi
    logging.error("OpenCV not found. Please install it using 'pip install opencv-python'. Camera processing will be limited.")
    cv2 = None

# Mengatur konfigurasi logging dasar agar output di konsol menjadi rapi dan informatif
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mengatur path sistem agar skrip ini bisa menemukan dan mengimpor modul custom dari folder lain
current_controller_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_controller_dir, "..", ".."))
sys.path.append(project_root)
controllers_dir = os.path.join(project_root, "controllers")
if controllers_dir not in sys.path:
    sys.path.append(controllers_dir)

# Mengimpor class DQNAgent dan fungsi evaluate_agent dari file-file terpisah
from dqn_agent.dqn_agent import DQNAgent
from Training_controller.Training_controller import evaluate_agent


class EPuckEnv:
    """Class ini membungkus robot E-Puck di Webots menjadi sebuah environment Reinforcement Learning yang standar."""
    
    def __init__(self, robot: Supervisor, track_id: int = 0):
        # --- Inisialisasi Dasar ---
        self.robot = robot
        self.timestep: int = int(self.robot.getBasicTimeStep())

        # --- Inisialisasi Perangkat Keras (Sensor & Motor) ---
        self.ground_sensor_names: list[str] = ['irR', 'irL', 'irGR', 'irGL', 'irCL', 'irCR']
        self.ground_sensors = {name: self.robot.getDevice(name) for name in self.ground_sensor_names}
        for sensor in self.ground_sensors.values(): sensor.enable(self.timestep)

        self.prox_sensor_names: list[str] = [f'ps{i}' for i in range(8)]
        self.prox_sensors = {name: self.robot.getDevice(name) for name in self.prox_sensor_names}
        for sensor in self.prox_sensors.values(): sensor.enable(self.timestep)
            
        self.lidar: Lidar = self.robot.getDevice('e_puck_lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud() 
            self.lidar_max_range: float = self.lidar.getMaxRange()
            self.lidar_num_beams: int = self.lidar.getHorizontalResolution()
            self.num_lidar_sectors: int = 8
        else:
            self.lidar_max_range, self.lidar_num_beams, self.num_lidar_sectors = 1.0, 360, 0

        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            self.target_camera_width = 64 
            self.target_camera_height = 64 
        else:
            self.camera_width, self.camera_height, self.target_camera_width, self.target_camera_height = 0, 0, 0, 0

        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)

        self.max_lives: int = 3
        self.lives: int = self.max_lives

        self.robot_node = self.robot.getSelf()
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.track_id: int = track_id
        self.checkpoints: dict[int, list[float]] = {0: [-1.89295, 1.61563, -0.0098853]}

        self.ground_calibration: dict[str, tuple[float, float]] = {name: (9.0, 1000.0) for name in self.ground_sensor_names}
        self.LINE_DETECT_THRESHOLD_NORMALIZED: float = 0.5   
        self.CENTER_LINE_DETECT_THRESHOLD_NORMALIZED: float = 0.8   
        self.max_proximity_value: float = 4096.0
        self.PROXIMITY_COLLISION_THRESHOLD_NORMALIZED: float = 0.7   

        self.ground_buffers: dict[str, list[float]] = {name: [] for name in self.ground_sensor_names}
        self.buffer_size: int = 5

        self.action_space_velocities: list[tuple[float, float]] = [(6.28, 6.28), (3.14, 6.28), (6.28, 3.14)]
        self.action_size: int = len(self.action_space_velocities)
        
        if self.camera:
            self.state_size: tuple = (self.target_camera_height, self.target_camera_width, 1)
        else:
            self.state_size: int = len(self.ground_sensor_names) + len(self.prox_sensor_names) + self.num_lidar_sectors

        self.REWARD_ON_TRACK_CENTER, self.REWARD_ON_TRACK_EDGE = 2.0, 0.5
        self.PENALTY_TIME_STEP, self.PENALTY_OFF_TRACK, self.PENALTY_COLLISION = -0.01, -3.0, -5.0
        self.REWARD_ACTION_STRAIGHT, self.REWARD_ACTION_TURN = 0.1, -0.05
        
        self.reset()

    def _normalize_ground_sensor(self, raw_value: float, sensor_name: str) -> float:
        min_val, max_val = self.ground_calibration[sensor_name]
        return max(0.0, min(1.0, (max_val - raw_value) / (max_val - min_val))) if (max_val - min_val) != 0 else 0.0

    def get_state(self) -> np.ndarray:
        if self.camera and cv2:
            camera_image_raw = self.camera.getImage()
            if camera_image_raw:
                img_array = np.frombuffer(camera_image_raw, np.uint8).reshape((self.camera_height, self.camera_width, 4))
                
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)
                img_resized = cv2.resize(img_gray, (self.target_camera_width, self.target_camera_height), interpolation=cv2.INTER_AREA)

                # --- PEMBARUAN TAMPILAN KAMERA ---
                display_size = (320, 320) # Ukuran jendela baru yang lebih besar
                display_img = cv2.resize(img_resized, display_size, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Tampilan Kamera Robot", display_img)
                cv2.waitKey(1)
                
                normalized_img = img_resized.astype(np.float32) / 255.0
                state_image = np.expand_dims(normalized_img, axis=-1)
                return state_image
            else:
                return np.zeros((self.target_camera_height, self.target_camera_width, 1), dtype=np.float32)
        elif self.camera and not cv2:
                logging.error("OpenCV not available. Cannot process camera image. Falling back to sensor data.")
        
        state_numeric: list[float] = []
        for name, sensor in self.ground_sensors.items():
            self.ground_buffers[name].append(sensor.getValue())
            if len(self.ground_buffers[name]) > self.buffer_size: self.ground_buffers[name].pop(0)
            state_numeric.append(self._normalize_ground_sensor(np.mean(self.ground_buffers[name]), name))

        for sensor in self.prox_sensors.values():
            state_numeric.append(max(0.0, min(1.0, sensor.getValue() / self.max_proximity_value)))

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
                    state_numeric.extend(lidar_features_normalized)
                else: state_numeric.extend([1.0] * self.num_lidar_sectors)
            else: state_numeric.extend([1.0] * self.num_lidar_sectors)
            
        return np.array(state_numeric, dtype=np.float32)

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
        return self.get_state()

    def teleport_to_checkpoint(self) -> np.ndarray:
        self._reset_simulation()
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        left_speed, right_speed = self.action_space_velocities[action]
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        if self.robot.step(self.timestep) == -1: 
            return np.zeros(self.state_size, dtype=np.float32), 0.0, True, {}
        
        next_state = self.get_state()
        reward, done, info = 0.0, False, {}
        reward += self.PENALTY_TIME_STEP

        action_reward_component = self.REWARD_ACTION_STRAIGHT if action == 0 else self.REWARD_ACTION_TURN
        reward += action_reward_component

        current_ground_values = [self._normalize_ground_sensor(sensor.getValue(), name) for name, sensor in self.ground_sensors.items()]
        current_prox_values = [max(0.0, min(1.0, sensor.getValue() / self.max_proximity_value)) for sensor in self.prox_sensors.values()]

        on_center = current_ground_values[4] > self.CENTER_LINE_DETECT_THRESHOLD_NORMALIZED and \
                    current_ground_values[5] > self.CENTER_LINE_DETECT_THRESHOLD_NORMALIZED
        on_track = any(s > self.LINE_DETECT_THRESHOLD_NORMALIZED for s in current_ground_values)

        track_reward_component = 0.0
        if on_center: track_reward_component = self.REWARD_ON_TRACK_CENTER
        elif on_track: track_reward_component = self.REWARD_ON_TRACK_EDGE
        else:
            track_reward_component = self.PENALTY_OFF_TRACK
            done, info['reason'] = True, 'Off Track'
        reward += track_reward_component

        obstacle_penalty_component = 0.0
        if any(val > self.PROXIMITY_COLLISION_THRESHOLD_NORMALIZED for val in current_prox_values):
            reward += self.PENALTY_COLLISION    
            obstacle_penalty_component = self.PENALTY_COLLISION
            self.lives -= 1
            if self.lives <= 0:
                done, info['reason'] = True, 'No Lives Left'
            else:
                self.teleport_to_checkpoint(); info['reason'] = 'Collision'
        
        info['reward_breakdown'] = {'track': track_reward_component, 'action': action_reward_component, 'obstacle': obstacle_penalty_component}
        return next_state, reward, done, info


# === BLOK EKSEKUSI UTAMA (MAIN SCRIPT EXECUTION BLOCK) ===
if __name__ == "__main__":
    robot = Supervisor()
    env = EPuckEnv(robot=robot, track_id=0)
    agent = DQNAgent(env.state_size, env.action_size)

    num_episodes = 900
    max_steps_per_episode = 1000
    batch_size = 32
    checkpoint_interval = 20

    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    logging.info(f"Memulai training untuk {num_episodes} episode.")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        info_breakdown = {'track': 0.0, 'action': 0.0, 'obstacle': 0.0}
        episode_losses = []
        episode_max_qs = []

        for step in range(max_steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

            if 'reward_breakdown' in info:
                for key in info_breakdown:
                    info_breakdown[key] += info['reward_breakdown'].get(key, 0.0)

            if done:
                break
        
        if len(agent.memory) > batch_size:
            replay_metrics = agent.replay()
            if replay_metrics:
                loss, avg_max_q = replay_metrics
                episode_losses.append(loss)
                episode_max_qs.append(avg_max_q)

        agent.update_epsilon(episode)

        avg_loss_episode = np.mean(episode_losses) if episode_losses else 0.0
        avg_max_q_episode = np.mean(episode_max_qs) if episode_max_qs else 0.0

        termination_reason = info.get('reason', 'Max Steps')
        logging.info(
            f"Episode {episode}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {step + 1} | "
            f"Epsilon: {agent.epsilon:.4f} | Avg Loss: {avg_loss_episode:.4f} | "
            f"Avg Max-Q: {avg_max_q_episode:.2f} | "
            f"Rewards (Track: {info_breakdown['track']:.2f}, Action: {info_breakdown['action']:.2f}, Obstacle: {info_breakdown['obstacle']:.2f}) | "
            f"Terminated by: {termination_reason}"
        )

        if episode % checkpoint_interval == 0:
            # --- PEMBARUAN: Menyimpan ke format .keras ---
            checkpoint_path = os.path.join(models_dir, f"dqn_epuck_episode_{episode}.keras")
            agent.set_save_path(checkpoint_path)
            agent.save_model()
            logging.info(f"--- Model Checkpoint disimpan di: {checkpoint_path} ---")

    # --- PEMBARUAN: Menyimpan model final ke format .keras ---
    final_model_path = os.path.join(models_dir, "dqn_epuck_final.keras")
    agent.set_save_path(final_model_path)
    agent.save_model()
    logging.info(f"Training selesai. Model final disimpan di: {final_model_path}")

    evaluate_agent(agent, env)