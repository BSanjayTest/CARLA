import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *

random.seed(78)

class SimEnv(object):
    def __init__(self, 
        visuals=True,
        target_speed = 30,
        max_iter = 4000,
        start_buffer = 10,
        train_freq = 1,
        save_freq = 200,
        start_ep = 0,
        max_dist_from_waypoint = 20,
        num_other_vehicles=3,
        num_pedestrians=10,
        weather_types=['Clear', 'Cloudy', 'Rain', 'Fog'],
        vehicle_types=['vehicle.nissan.patrol', 'vehicle.tesla.model3', 'vehicle.ford.mustang', 'vehicle.bmw.isetta']
    ) -> None:
        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0) # Increased timeout

        # Connect to world and ensure it's in a responsive state
        max_retries = 3
        retry_count = 0
        connected = False
        
        while not connected and retry_count < max_retries:
            try:
                self.world = self.client.get_world()
                
                # IMPORTANT: Disable synchronous mode FIRST before anything else
                # This is the most common cause of timeouts and crashes
                print(f"Connection attempt {retry_count+1}: Ensuring asynchronous mode...")
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)
                self.world.tick() # Apply settings
                
                # Aggressive global cleanup of ALL actors from previous runs
                print("Cleaning up world from previous sessions...")
                for actor in self.world.get_actors():
                    if any(x in actor.type_id for x in ['vehicle', 'sensor', 'walker', 'controller']):
                        try:
                            if actor.is_alive:
                                actor.destroy()
                        except:
                            pass
                
                # If current world is not the desired one, load it
                if self.world.get_map().name != 'Town02_Opt':
                    print("Loading Town02_Opt...")
                    self.world = self.client.load_world('Town02_Opt')
                
                connected = True
            except Exception as e:
                retry_count += 1
                print(f"Connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    print("Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    print("Could not connect to CARLA after multiple attempts. Please ensure the simulator is running.")
                    raise e

        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        

        self.spawn_points = self.world.get_map().get_spawn_points()

        # Enhanced environment parameters - assign before using
        self.num_other_vehicles = num_other_vehicles
        self.num_pedestrians = num_pedestrians
        self.weather_types = weather_types
        self.current_weather = None
        self.other_vehicles = []
        self.npc_controllers = []
        self.walkers = []
        self.walker_controllers = []

        self.blueprint_library = self.world.get_blueprint_library()
        
        # Validate vehicle blueprints to prevent IndexError
        available_blueprints = [bp.id for bp in self.blueprint_library.filter('vehicle.*')]
        self.vehicle_types = [vt for vt in vehicle_types if vt in available_blueprints]
        
        if not self.vehicle_types:
            # Fallback to something that MUST exist
            self.vehicle_types = ['vehicle.tesla.model3']
            print("Warning: None of the configured vehicle types were found. Falling back to vehicle.tesla.model3")
        
        self.vehicle_blueprint = self.blueprint_library.find(random.choice(self.vehicle_types))

        # input these later on as arguments
        self.global_t = 0 # global timestep
        self.target_speed = target_speed # km/h 
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.start_ep = start_ep

        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer
        
        self.total_rewards = 0
        self.average_rewards_list = []
        
        # Telemetry Logging
        self.log_file = 'training_log.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('episode,steps,reward,collisions,weather,avg_speed,lane_distance,lat,lon,accel_x,gyro_z\n')
    
    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
    
    def create_actors(self):
        self.actor_list = []
        self.walkers = []
        self.walker_controllers = []
        
        # Set random weather conditions
        self._set_weather_conditions()
        
        # Spawn main vehicle with random type
        self.vehicle_blueprint = self.blueprint_library.find(random.choice(self.vehicle_types))
        
        # Try spawning the main vehicle at different locations until success
        spawn_success = False
        max_spawn_attempts = 10
        attempt = 0
        
        while not spawn_success and attempt < max_spawn_attempts:
            self.main_spawn_point = random.choice(self.spawn_points)
            self.vehicle = self.world.try_spawn_actor(self.vehicle_blueprint, self.main_spawn_point)
            if self.vehicle is not None:
                spawn_success = True
                print(f"Main vehicle spawned successfully on attempt {attempt+1}")
            else:
                attempt += 1
                print(f"Spawn attempt {attempt} failed, trying another location...")
        
        if not spawn_success:
            # Absolute fallback if all random attempts fail
            self.main_spawn_point = self.spawn_points[0]
            self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, self.main_spawn_point)
            print("Forced spawn at first spawn point after all random attempts failed")

        self.actor_list.append(self.vehicle)
        
        # Spawn other vehicles (actors)
        self._spawn_other_vehicles()
        
        # Spawn pedestrians (walkers)
        self._spawn_pedestrians()

        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        # Third-person chase camera with better view
        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-8, z=4), carla.Rotation(pitch=-20)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        # Top-down bird's eye view camera
        self.camera_top = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=0, z=12), carla.Rotation(pitch=-90)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_top)

        # Side camera
        self.camera_side = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=0, y=-10, z=2), carla.Rotation(yaw=-90)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_side)

        # Radar sensor for distance detection
        self.radar = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.radar'),
            carla.Transform(carla.Location(x=2.8, z=1.0), carla.Rotation(pitch=5)),
            attach_to=self.vehicle)
        self.actor_list.append(self.radar)

        # Semantic segmentation camera for rule-based perception
        self.camera_sem = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_sem)

        # Depth camera for distance visualization
        self.camera_depth = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.depth'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_depth)

        # Lidar sensor for 3D point cloud
        self.lidar = self.world.spawn_actor(
            self.blueprint_library.find('sensor.lidar.ray_cast'),
            carla.Transform(carla.Location(x=0, z=2.4)),
            attach_to=self.vehicle)
        self.actor_list.append(self.lidar)

        # GNSS (GPS) sensor for global positioning
        self.gnss = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.gnss'),
            carla.Transform(carla.Location(x=1.0, z=2.4)),
            attach_to=self.vehicle)
        self.actor_list.append(self.gnss)

        # IMU sensor for acceleration and angular velocity
        self.imu = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.imu'),
            carla.Transform(carla.Location(x=0, z=2.4)),
            attach_to=self.vehicle)
        self.actor_list.append(self.imu)

        # Lane Invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion_sensor)

        # Obstacle detection sensor
        self.obstacle_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.obstacle'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_sensor)

        # Current camera mode (1=front, 2=third-person, 3=top, 4=side, 5=NPC tracking, 6=Semantic, 7=Depth, 8=Lidar, 9=All)
        self.current_camera_mode = 2
        self.current_npc_index = 0
        
        # NPC tracking camera - we spawn it once and will re-attach/move it
        self.camera_npc_follow = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-8, z=4), carla.Rotation(pitch=-20)),
            attach_to=self.vehicle) # Attach to something for now
        self.actor_list.append(self.camera_npc_follow)


        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)

        self.speed_controller = PIDLongitudinalController(self.vehicle)
    
    def _set_weather_conditions(self):
        """Set random weather conditions for training diversity"""
        weather_type = random.choice(self.weather_types)
        weather = self.world.get_weather()
        
        if weather_type == 'Clear':
            weather.cloudiness = 0.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 0.0
            weather.fog_density = 0.0
        elif weather_type == 'Cloudy':
            weather.cloudiness = 80.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 30.0
            weather.fog_density = 0.0
        elif weather_type == 'Rain':
            weather.cloudiness = 100.0
            weather.precipitation = 80.0
            weather.precitation_deposits = 50.0
            weather.wind_intensity = 60.0
            weather.fog_density = 10.0
        elif weather_type == 'Fog':
            weather.cloudiness = 50.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 20.0
            weather.fog_density = 70.0
        
        self.world.set_weather(weather)
        self.current_weather = weather_type
        print(f"Weather set to: {weather_type}")
    
    def _spawn_other_vehicles(self):
        """Spawn NPC vehicles with different behaviors"""
        self.other_vehicles = []
        self.npc_controllers = []
        
        # Get available spawn points
        available_spawn_points = self.spawn_points.copy()
        # Find the point used by main vehicle and remove it
        # Since it's a list of transforms, we can remove the stored object
        if hasattr(self, 'main_spawn_point'):
            # Using list comprehension or a loop to find match in case it's not the exact object reference
            available_spawn_points = [sp for sp in available_spawn_points 
                                     if sp.location.distance(self.main_spawn_point.location) > 1.0]
        
        for i in range(self.num_other_vehicles):
            if not available_spawn_points:
                break
                
            # Choose random vehicle type
            vehicle_bp = self.blueprint_library.find(random.choice(self.vehicle_types))
            spawn_point = random.choice(available_spawn_points)
            available_spawn_points.remove(spawn_point)
            
            try:
                npc_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.other_vehicles.append(npc_vehicle)
                self.actor_list.append(npc_vehicle)
                
                # Set random behavior for NPC
                behavior_type = random.choice(['aggressive', 'normal', 'cautious'])
                self._set_npc_behavior(npc_vehicle, behavior_type)
                
                print(f"Spawned NPC vehicle {i+1} with {behavior_type} behavior")
                
            except Exception as e:
                print(f"Failed to spawn NPC vehicle {i+1}: {e}")
    
    def _spawn_pedestrians(self):
        """Spawn pedestrians (walkers) in the environment"""
        # Get walker blueprints
        walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
        
        for i in range(self.num_pedestrians):
            try:
                # Pick a random point on the sidewalk
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                else:
                    # Fallback to random spawn point location
                    spawn_point.location = random.choice(self.spawn_points).location
                
                # Choose random walker type
                walker_bp = random.choice(walker_blueprints)
                
                # Spawn walker
                walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                if walker:
                    # Spawn controller
                    walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
                    walker_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                    
                    # Start walker
                    walker_controller.start()
                    walker_controller.go_to_location(self.world.get_random_location_from_navigation())
                    walker_controller.set_max_speed(1 + random.random()) # Random speed between 1-2 m/s
                    
                    self.walkers.append(walker)
                    self.walker_controllers.append(walker_controller)
                    self.actor_list.append(walker)
                    self.actor_list.append(walker_controller)
                    
                    print(f"Spawned pedestrian {i+1}")
            except Exception as e:
                print(f"Failed to spawn pedestrian {i+1}: {e}")

    def _set_npc_behavior(self, vehicle, behavior_type):
        """Set different driving behaviors for NPC vehicles"""
        if behavior_type == 'aggressive':
            # Fast and aggressive driving
            vehicle.set_autopilot(True)
            # Configure traffic manager for aggressive behavior
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.vehicle_percentage_speed_difference(vehicle, -20)  # 20% faster
            traffic_manager.distance_to_leading_vehicle(vehicle, 2)  # Close following distance
            traffic_manager.vehicle_lane_offset(vehicle, 0.3)  # Lane weaving
            
        elif behavior_type == 'normal':
            # Normal driving behavior
            vehicle.set_autopilot(True)
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)  # Normal speed
            traffic_manager.distance_to_leading_vehicle(vehicle, 5)  # Normal following distance
            
        elif behavior_type == 'cautious':
            # Slow and cautious driving
            vehicle.set_autopilot(True)
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 30)  # 30% slower
            traffic_manager.distance_to_leading_vehicle(vehicle, 10)  # Large following distance
    
    def reset(self):
        """Enhanced reset with proper cleanup of NPC vehicles, walkers and weather"""
        # Stop and clean up walker controllers
        for controller in self.walker_controllers:
            try:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
            except RuntimeError:
                pass
        
        # Explicitly destroy all sensors to avoid orphaned sensor limits
        sensor_blueprints = [
            'sensor.camera.rgb', 
            'sensor.other.collision', 
            'sensor.other.radar', 
            'sensor.camera.semantic_segmentation',
            'sensor.camera.depth',
            'sensor.lidar.ray_cast',
            'sensor.other.gnss',
            'sensor.other.imu',
            'sensor.other.lane_invasion',
            'sensor.other.obstacle'
        ]
        for actor in self.world.get_actors():
            if actor.type_id in sensor_blueprints:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except RuntimeError:
                    pass
        
        # Clean up other vehicles
        for vehicle in self.other_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
            except RuntimeError:
                pass
        
        # Clean up main actors
        for actor in self.actor_list:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        
        self.actor_list = []
        self.other_vehicles = []
        self.npc_controllers = []
        self.walkers = []
        self.walker_controllers = []
        
        # Small delay to ensure proper cleanup - reduced from 0.1
        import time
        time.sleep(0.02)
    
    def change_weather_during_episode(self):
        """Dynamically change weather during training for more challenging scenarios"""
        if random.random() < 0.1:  # 10% chance to change weather during episode
            self._set_weather_conditions()
    
    def _switch_npc_camera(self):
        """Cycle through NPC vehicles for tracking"""
        if not self.other_vehicles:
            print("No other vehicles to track.")
            return
            
        self.current_npc_index = (self.current_npc_index + 1) % len(self.other_vehicles)
        target_npc = self.other_vehicles[self.current_npc_index]
        npc_type = target_npc.type_id.split('.')[-1].replace('_', ' ').title()
        
        print(f"Now tracking NPC {self.current_npc_index + 1}/{len(self.other_vehicles)}: {npc_type}")

    def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
        self.total_rewards = 0
        total_speed = 0
        total_dist = 0
        
        # Build sensors list - now including all available sensors
        sensors = [
            self.camera_rgb, 
            self.camera_rgb_vis, 
            self.camera_top, 
            self.camera_side, 
            self.camera_npc_follow, 
            self.camera_sem, 
            self.camera_depth,
            self.lidar, 
            self.gnss,
            self.imu,
            self.radar
        ]
        
        # Note: Lane Invasion and Obstacle sensors are event-based and handled via callbacks if needed, 
        # but for now we focus on the data sensors for the sync loop.
        
        with CarlaSyncMode(self.world, *sensors, self.collision_sensor, fps=30) as sync_mode:
            counter = 0

            # Check initial tick with reduced timeout for faster startup
            tick_data = sync_mode.tick(timeout=0.2)
            if not tick_data:
                return True
                
            snapshot, image_rgb, image_rgb_vis, image_top, image_side, image_npc, image_sem, image_depth, lidar_data, gnss_data, imu_data, radar_data, collision_event = tick_data

            collision = 1 if collision_event else 0

            vehicle_location = self.vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, 
                lane_type=carla.LaneType.Driving)

            cos_yaw_diff, dist, _, current_speed = get_reward_comp(self.vehicle, waypoint, collision)
            reward = reward_value(cos_yaw_diff, dist, collision, current_speed, self.target_speed)

            # destroy if there is no data
            if snapshot is None or image_rgb is None:
                print("No data, skipping episode")
                self.reset()
                return True

            image = process_img(image_rgb)
            next_state = image 

            while True:
                if self.visuals:
                    # Check for camera switching
                    camera_switch = check_camera_switch()
                    if camera_switch == -1:  # Quit
                        return False
                    elif camera_switch > 0:  # Switch camera
                        self.current_camera_mode = camera_switch
                        if camera_switch == 5:
                            self._switch_npc_camera()
                        print(f"Switched to camera mode {camera_switch}")
                    
                    self.clock.tick_busy_loop(30)

                vehicle_location = self.vehicle.get_location()

                waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, 
                    lane_type=carla.LaneType.Driving)
                
                speed = get_speed(self.vehicle)
                total_speed += speed
                total_dist += dist

                # 1. Perception: Check for nearby pedestrians and obstacles via Radar
                # We'll calculate a "safety score" or "proximity penalty"
                min_dist = 100.0
                if radar_data:
                    for detect in radar_data:
                        # detect.depth is distance to obstacle
                        if detect.depth < min_dist:
                            min_dist = detect.depth
                
                # 2. Perception: Traffic Light Detection
                traffic_light_state = self.vehicle.get_traffic_light_state()
                is_red_light = False
                if self.vehicle.is_at_traffic_light():
                    if traffic_light_state == carla.TrafficLightState.Red:
                        is_red_light = True
                    elif traffic_light_state == carla.TrafficLightState.Yellow:
                        is_red_light = True # Treat yellow as red for safe college project driving
                
                # Rule: Stop if pedestrian or obstacle is too close (e.g., < 5 meters) OR Red light
                emergency_stop = False
                if min_dist < 5.0 or is_red_light:
                    emergency_stop = True

                # Advance the simulation and wait for the data.
                state = next_state

                counter += 1
                self.global_t += 1


                action = model.select_action(state, eval=eval)
                steer = action
                if action_map is not None:
                    steer = action_map[action]

                # Control: Apply emergency stop if needed
                if emergency_stop:
                    control = carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0)
                else:
                    control = self.speed_controller.run_step(self.target_speed)
                    control.steer = steer
                
                self.vehicle.apply_control(control)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Move NPC tracking camera if in mode 5
                if self.current_camera_mode == 5 and self.other_vehicles:
                    target_npc = self.other_vehicles[self.current_npc_index]
                    if target_npc.is_alive:
                        # Get NPC transform and offset it for chase view
                        npc_trans = target_npc.get_transform()
                        
                        # Calculate offset position behind NPC
                        import math
                        yaw_rad = math.radians(npc_trans.rotation.yaw)
                        offset_x = -8.0 * math.cos(yaw_rad)
                        offset_y = -8.0 * math.sin(yaw_rad)
                        
                        follow_trans = carla.Transform(
                            carla.Location(
                                x=npc_trans.location.x + offset_x,
                                y=npc_trans.location.y + offset_y,
                                z=npc_trans.location.z + 4.0
                            ),
                            carla.Rotation(pitch=-20, yaw=npc_trans.rotation.yaw)
                        )
                        self.camera_npc_follow.set_transform(follow_trans)

                tick_data = sync_mode.tick(timeout=0.1)
                if not tick_data:
                    print("Tick timeout - checking for immediate crash...")
                    # Check if vehicle is still alive - if not, it's a crash
                    if not self.vehicle.is_alive:
                        collision = 1
                        done = 1
                        # Log immediately on crash
                        avg_speed = total_speed / counter if counter > 0 else 0
                        avg_lane_dist = total_dist / counter if counter > 0 else 0
                        try:
                            final_lat = gnss_data.latitude if gnss_data else 0.0
                            final_lon = gnss_data.longitude if gnss_data else 0.0
                            final_accel_x = imu_data.accelerometer.x if imu_data else 0.0
                            final_gyro_z = imu_data.gyroscope.z if imu_data else 0.0
                        except:
                            final_lat, final_lon, final_accel_x, final_gyro_z = 0.0, 0.0, 0.0, 0.0
                        with open(self.log_file, 'a') as f:
                            f.write(f'{self.start_ep + ep},{counter},{self.total_rewards:.2f},{collision},{self.current_weather},{avg_speed:.2f},{avg_lane_dist:.2f},{final_lat:.6f},{final_lon:.6f},{final_accel_x:.2f},{final_gyro_z:.2f}\n')
                        print(f"COLLISION DETECTED (immediate)! Resetting now...")
                    break
                    
                snapshot, image_rgb, image_rgb_vis, image_top, image_side, image_npc, image_sem, image_depth, lidar_data, gnss_data, imu_data, radar_data, collision_event = tick_data

                collision = 1 if collision_event else 0
                is_pedestrian_collision = False
                if collision_event:
                    # Check if the hit actor is a walker
                    if 'walker' in collision_event.other_actor.type_id:
                        is_pedestrian_collision = True
                        print("PEDESTRIAN COLLISION DETECTED!")

                cos_yaw_diff, dist, _, current_speed = get_reward_comp(self.vehicle, waypoint, collision)
                
                # Custom reward based on proximity to pedestrians/obstacles
                proximity_penalty = 0.0
                if min_dist < 10.0:
                    proximity_penalty = -10.0 * (1.0 - min_dist/10.0)
                
                # Huge penalty for hitting a pedestrian
                pedestrian_penalty = -500.0 if is_pedestrian_collision else 0.0

                # Traffic Light Penalty (Running a red light)
                light_penalty = 0.0
                if is_red_light and current_speed > 2.0:
                    light_penalty = -20.0 # Penalty for moving while light is red

                reward = reward_value(cos_yaw_diff, dist, collision, current_speed, self.target_speed) + \
                         proximity_penalty + pedestrian_penalty + light_penalty

                if snapshot is None or image_rgb is None:
                    print("Process ended here")
                    break

                image = process_img(image_rgb)

                # Immediate respawn on ANY collision
                done = 1 if collision else 0

                self.total_rewards += reward

                next_state = image

                replay_buffer.add(state, action, next_state, reward, done)

                if not eval:
                    if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        model.train(replay_buffer)

                # Draw the display.
                if self.visuals:
                    # Select camera based on current mode
                    display_image = None
                    camera_name = ""
                    
                    if self.current_camera_mode == 1:
                        display_image = image_rgb
                        camera_name = "Front Camera (AI View)"
                    elif self.current_camera_mode == 2:
                        display_image = image_rgb_vis
                        camera_name = "Third-Person View"
                    elif self.current_camera_mode == 3:
                        display_image = image_top
                        camera_name = "Top-Down View"
                    elif self.current_camera_mode == 4:
                        display_image = image_side
                        camera_name = "Side View"
                    elif self.current_camera_mode == 5:
                        display_image = image_npc
                        # Get NPC vehicle name for clarity
                        npc_name = "Unknown"
                        if self.other_vehicles and self.current_npc_index < len(self.other_vehicles):
                            npc_name = self.other_vehicles[self.current_npc_index].type_id.split('.')[-1].replace('_', ' ').title()
                        camera_name = f"FOLLOWING: {npc_name} ({self.current_npc_index + 1}/{len(self.other_vehicles)})"
                        
                        # Add a special warning box for NPC mode
                        pygame.draw.rect(self.display, (255, 0, 0), (250, 10, 300, 30))
                        self.display.blit(self.font.render('--- NPC VIEW (NOT THE AGENT) ---', True, (255, 255, 255)), (265, 15))
                        self.display.blit(self.font.render('Press 5 to switch NPC, 1-4 to return', True, (255, 255, 255)), (265, 45))
                    elif self.current_camera_mode == 6:
                        # Process semantic image for display (colorizing it)
                        image_sem.convert(carla.ColorConverter.CityScapesPalette)
                        display_image = image_sem
                        camera_name = "Semantic Segmentation"
                    elif self.current_camera_mode == 7:
                        # Process depth image for display (grayscale depth)
                        image_depth.convert(carla.ColorConverter.LogarithmicDepth)
                        display_image = image_depth
                        camera_name = "Depth View"
                    elif self.current_camera_mode == 8:
                        # Visualize Lidar Point Cloud - ENHANCED VERSION
                        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
                        points = np.reshape(points, (int(points.shape[0] / 4), 4))
                        
                        # Create a black background surface
                        lidar_surface = pygame.Surface((800, 600))
                        lidar_surface.fill((0, 0, 0))
                        
                        # Draw distance rings (10m, 20m, 30m)
                        for radius in [10, 20, 30]:
                            pygame.draw.circle(lidar_surface, (40, 40, 40), (400, 300), radius * 10, 1)
                            self.display.blit(self.font.render(f'{radius}m', True, (60, 60, 60)), (400 + radius * 10, 300))

                        # Draw crosshair axes
                        pygame.draw.line(lidar_surface, (30, 30, 30), (0, 300), (800, 300), 1)
                        pygame.draw.line(lidar_surface, (30, 30, 30), (400, 0), (400, 600), 1)

                        # Project points to 2D with color based on intensity/height
                        for p in points:
                            # p[0]=x, p[1]=y, p[2]=z, p[3]=intensity
                            # Scale: 1 meter = 10 pixels
                            x, y = int(400 + p[1] * 10), int(300 - p[0] * 10)
                            
                            if 0 <= x < 800 and 0 <= y < 600:
                                # Color based on height (z)
                                z_height = p[2]
                                if z_height > 1.0: color = (255, 255, 255) # High objects (white)
                                elif z_height > -0.5: color = (0, 255, 0)   # Mid objects (green)
                                else: color = (0, 100, 0)                  # Ground (dark green)
                                
                                lidar_surface.set_at((x, y), color)
                        
                        # Draw ego-vehicle icon (a blue triangle) at the center
                        pygame.draw.polygon(lidar_surface, (0, 100, 255), [(395, 305), (405, 305), (400, 290)])
                        
                        display_image_surface = lidar_surface
                        camera_name = "Top-Down Lidar Scan"
                    elif self.current_camera_mode == 9:
                        # Show all sensors status summary
                        display_image = image_rgb_vis
                        camera_name = "Full Sensor Suite Status"
                    
                    if self.current_camera_mode == 8:
                        self.display.blit(display_image_surface, (0, 0))
                    elif display_image:
                        draw_image(self.display, display_image)
                    
                    # Display camera info and controls
                    self.display.blit(
                        self.font.render(f'Camera: {camera_name}', True, (255, 255, 0)),
                        (8, 10))
                    self.display.blit(
                        self.font.render('Controls: 1-4=Cam, 5=NPC, 6=Sem, 7=Depth, 8=Lidar, 9=GPS/IMU', True, (255, 255, 255)),
                        (8, 25))
                    self.display.blit(
                        self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                        (8, 40))
                    self.display.blit(
                        self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                        (8, 55))
                    self.display.blit(
                        self.font.render('Min Distance to Obstacle: %.2f m' % min_dist, True, (255, 0, 0) if min_dist < 10 else (0, 255, 0)),
                        (8, 70))
                    
                    # AI DECISION DASHBOARD
                    # Background for the dashboard
                    pygame.draw.rect(self.display, (0, 0, 0, 150), (600, 10, 190, 170))
                    pygame.draw.rect(self.display, (255, 255, 255), (600, 10, 190, 170), 1)
                    
                    self.display.blit(self.font.render('AI DECISION HUB', True, (0, 255, 255)), (610, 15))
                    
                    # Steering Status
                    steer_color = (0, 255, 0) if abs(steer) < 0.1 else (255, 255, 0)
                    if abs(steer) > 0.5: steer_color = (255, 0, 0)
                    self.display.blit(self.font.render(f'Steer: {steer:.2f}', True, steer_color), (610, 35))
                    
                    # Throttle/Brake Status
                    status_text = "CRUISING"
                    status_color = (0, 255, 0)
                    if emergency_stop:
                        status_text = "BRAKING (SAFE)"
                        if is_red_light:
                            status_text = "RED LIGHT STOP"
                        elif min_dist < 5.0:
                            status_text = "OBSTACLE DETECTED"
                        status_color = (255, 0, 0)
                    elif current_speed < 5:
                        status_text = "STARTING"
                    
                    self.display.blit(self.font.render(f'Status: {status_text}', True, status_color), (610, 55))
                    self.display.blit(self.font.render(f'Speed: {current_speed:.1f} km/h', True, (255, 255, 255)), (610, 75))
                    
                    # Traffic Light Visual
                    light_color = (0, 255, 0) if traffic_light_state == carla.TrafficLightState.Green else (255, 0, 0)
                    if traffic_light_state == carla.TrafficLightState.Yellow: light_color = (255, 255, 0)
                    self.display.blit(self.font.render(f'Light: {traffic_light_state}', True, light_color), (610, 95))

                    # Perception Status
                    alignment = (cos_yaw_diff + 1) / 2 * 100 # Normalize to 0-100%
                    self.display.blit(self.font.render(f'Lane Align: {alignment:.1f}%', True, (255, 255, 255)), (610, 120))
                    self.display.blit(self.font.render(f'Lane Dist: {dist:.2f} m', True, (255, 255, 255)), (610, 140))

                    if self.current_camera_mode == 9:
                        # Extra Dashboard for Mode 9 (GPS/IMU Data)
                        pygame.draw.rect(self.display, (0, 0, 0, 150), (10, 100, 220, 120))
                        pygame.draw.rect(self.display, (255, 255, 255), (10, 100, 220, 120), 1)
                        self.display.blit(self.font.render('LOCATION & INERTIA', True, (255, 255, 0)), (20, 105))
                        self.display.blit(self.font.render(f'Lat: {gnss_data.latitude:.6f}', True, (255, 255, 255)), (20, 125))
                        self.display.blit(self.font.render(f'Lon: {gnss_data.longitude:.6f}', True, (255, 255, 255)), (20, 145))
                        self.display.blit(self.font.render(f'Accel X: {imu_data.accelerometer.x:.2f} m/s^2', True, (255, 255, 255)), (20, 165))
                        self.display.blit(self.font.render(f'Gyro Z: {imu_data.gyroscope.z:.2f} rad/s', True, (255, 255, 255)), (20, 185))

                    pygame.display.flip()

                # Dynamic weather changes during episode
                if counter % 100 == 0:  # Check every 100 steps
                    self.change_weather_during_episode()
                
                # Check for collision IMMEDIATELY - no waiting
                if collision == 1:
                    # Log IMMEDIATELY on crash detection
                    avg_speed = total_speed / counter if counter > 0 else 0
                    avg_lane_dist = total_dist / counter if counter > 0 else 0
                    try:
                        final_lat = gnss_data.latitude if gnss_data else 0.0
                        final_lon = gnss_data.longitude if gnss_data else 0.0
                        final_accel_x = imu_data.accelerometer.x if imu_data else 0.0
                        final_gyro_z = imu_data.gyroscope.z if imu_data else 0.0
                    except:
                        final_lat, final_lon, final_accel_x, final_gyro_z = 0.0, 0.0, 0.0, 0.0
                    with open(self.log_file, 'a') as f:
                        f.write(f'{self.start_ep + ep},{counter},{self.total_rewards:.2f},{collision},{self.current_weather},{avg_speed:.2f},{avg_lane_dist:.2f},{final_lat:.6f},{final_lon:.6f},{final_accel_x:.2f},{final_gyro_z:.2f}\n')
                    print(f"COLLISION DETECTED! Resetting immediately...")
                    print(f"Episode {ep} completed - Steps: {counter}, Reward: {self.total_rewards:.2f}, Collision: {collision}")
                    break
                
                # Check other termination conditions
                if counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    # Logging stats to CSV
                    avg_speed = total_speed / counter if counter > 0 else 0
                    avg_lane_dist = total_dist / counter if counter > 0 else 0
                    
                    # Get final GNSS and IMU readings with safe access
                    try:
                        final_lat = gnss_data.latitude if gnss_data else 0.0
                        final_lon = gnss_data.longitude if gnss_data else 0.0
                        final_accel_x = imu_data.accelerometer.x if imu_data else 0.0
                        final_gyro_z = imu_data.gyroscope.z if imu_data else 0.0
                    except:
                        final_lat, final_lon, final_accel_x, final_gyro_z = 0.0, 0.0, 0.0, 0.0

                    with open(self.log_file, 'a') as f:
                        f.write(f'{self.start_ep + ep},{counter},{self.total_rewards:.2f},{collision},{self.current_weather},{avg_speed:.2f},{avg_lane_dist:.2f},{final_lat:.6f},{final_lon:.6f},{final_accel_x:.2f},{final_gyro_z:.2f}\n')

                    print(f"Episode {ep} completed - Steps: {counter}, Reward: {self.total_rewards:.2f}, Collision: {collision}, Weather: {self.current_weather}")
                    break
            
            if ep % self.save_freq == 0 and ep > 0:
                self.save(model, ep)
            
            return True
    
    def save(self, model, ep):
        """Save model checkpoint"""
        model.save(f'weights/model_ep_{ep}')
        print(f"Model saved at episode {ep}")
    
    def quit(self):
        """Clean shutdown - leaves simulator in a 'like new' state"""
        # 1. First, disable synchronous mode immediately so the simulator doesn't freeze
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)
            print("Synchronous mode disabled. Simulator is now responsive.")
            
            # Tick once to apply the mode change
            self.world.tick()
        except Exception as e:
            print(f"Error while disabling synchronous mode: {e}")
            
        # 2. Aggressively clean up all actors we might have spawned
        try:
            print("Cleaning up sensors, vehicles and walkers...")
            # We iterate through all actors in the world to be 100% sure nothing is left
            all_actors = self.world.get_actors()
            for actor in all_actors:
                if any(x in actor.type_id for x in ['vehicle', 'sensor', 'walker', 'controller']):
                    try:
                        if actor.is_alive:
                            actor.destroy()
                    except:
                        pass
            
            # Perform regular reset as well
            self.reset()
        except Exception as e:
            print(f"Error during actor cleanup: {e}")

        # 3. Reload the world - this is the ONLY way to guarantee a 'like new' state
        try:
            print("Reloading Town02_Opt for a fresh simulator state...")
            self.client.load_world('Town02_Opt')
            print("Simulator reset to 'like new' state successfully.")
        except Exception as e:
            print(f"Warning: Could not reload world on exit: {e}")
            
        if self.visuals:
            try:
                pygame.quit()
            except:
                pass