# Multi-Sensor Autonomous Driving Agent (College Final Year Project)

This project implements a state-of-the-art autonomous driving system using **Deep Reinforcement Learning (DQN)** and **Sensor Fusion** within the CARLA simulator.

## 🚀 Key Features
- **Deep Reinforcement Learning**: Powered by a DQN model that learns optimal driving policies through rewards and penalties.
- **Sensor Fusion**: Integrated suite of 13+ sensors for comprehensive environmental awareness.
- **Hybrid Safety System**: Rule-based safety overrides for emergency braking and traffic light compliance.
- **Real-Time Analytics**: Live "AI Decision Hub" HUD showing steering, speed, and perception data.
- **Telemetry Logging**: Automatic CSV logging of training metrics for data analysis and graphing.

## 📡 Sensor Suite
The vehicle is equipped with a professional-grade sensor array:
- **Cameras**: RGB (Front, Side, Top, Third-Person), Semantic Segmentation (color-coded object detection), and Depth (distance mapping).
- **Lidar**: 3D Ray-Cast Lidar for point cloud generation.
- **Radar**: Long-range obstacle detection and speed estimation.
- **Localization**: GNSS (GPS) and IMU (Accelerometer/Gyroscope) for precise positioning and motion tracking.
- **Events**: Collision, Lane Invasion, and Proximity sensors.

## 🎮 Controls (During Simulation)
Press the number keys to switch between different sensor views:
- `1-4`: Standard Agent Views (Front, Third-Person, Top, Side)
- `5`: NPC Tracking Mode (Follow other vehicles)
- `6`: **Semantic Segmentation View**
- `7`: **Depth View**
- `8`: **Lidar Point Cloud View**
- `9`: **Full Sensor Suite (GPS/IMU Data Overlay)**
- `ESC`: Clean Quit & Simulator Reset

## 🛠️ Getting Started
1. **Prerequisites**: Ensure CARLA Simulator is running.
2. **Training**: Run `python train.py` to start the reinforcement learning process.
3. **Evaluation**: Run `python main.py` to test the agent using the latest trained weights.
4. **Cleanup**: Run `python force_cleanup.py` if the simulator becomes unresponsive.

## � Data Analysis
All training data is saved to `training_log.csv`. This data can be used to generate graphs for:
- Reward vs. Episode (Proof of Learning)
- Average Lane Distance (Path Accuracy)
- Collision Frequency (Safety Improvement)

## 💻 Hardware & Performance
This project is designed to be accessible on modern consumer laptops (e.g., 16GB RAM, 6GB+ VRAM GPU).

### **Minimum Requirements**
- **OS**: Windows 10/11 or Ubuntu 20.04+.
- **RAM**: 16GB (8GB might experience stuttering).
- **GPU**: NVIDIA RTX 2060 or higher (6GB VRAM minimum for full sensor suite).
- **Storage**: ~30GB for CARLA Simulator and map assets.

### **Optimization for 16GB RAM / MacBook**
If you are running on a 16GB RAM laptop or an Apple Silicon MacBook (M1/M2/M3), use these tips to maintain high FPS:
1. **Low Quality Mode**: Run CARLA with `-quality-level=Low` to significantly reduce VRAM and RAM usage.
2. **Reduce Sensor Load**: In `config.py`, reduce `num_other_vehicles` to 5 and `num_pedestrians` to 0 to lower CPU overhead.
3. **Resolution**: The Pygame window is set to 800x600; keeping this small helps performance.
4. **MacBook Note**: CARLA is primarily developed for Windows/Linux. For Mac, it is recommended to run CARLA inside a **Docker container** or use the experimental Mac builds if available.

---
*Developed as a Final Year Engineering Project.*
