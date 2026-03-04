import os
import cv2
import pygame
import math
import numpy as np

def process_img(image, dim_x=128, dim_y=128):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # scale_percent = 25
    # width = int(array.shape[1] * scale_percent/100)
    # height = int(array.shape[0] * scale_percent/100)

    # dim = (width, height)
    dim = (dim_x, dim_y)  # set same dim for now
    resized_img = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    scaledImg = img_gray/255.

    # normalize
    mean, std = 0.5, 0.5
    normalizedImg = (scaledImg - mean) / std

    return normalizedImg

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def check_camera_switch():
    """
    Check for camera switching keyboard input.
    Returns:
        -1 for quit (ESC)
        1-9 for camera mode switching
        0 for no action
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return -1
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return -1
            elif event.key == pygame.K_1:
                return 1  # Front camera
            elif event.key == pygame.K_2:
                return 2  # Third-person view
            elif event.key == pygame.K_3:
                return 3  # Top-down view
            elif event.key == pygame.K_4:
                return 4  # Side view
            elif event.key == pygame.K_5:
                return 5  # NPC tracking view
            elif event.key == pygame.K_6:
                return 6  # Semantic view
            elif event.key == pygame.K_7:
                return 7  # Depth view
            elif event.key == pygame.K_8:
                return 8  # Lidar view
            elif event.key == pygame.K_9:
                return 9  # All Sensors view
    return 0

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.
        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def correct_yaw(x):
    return(((x%360) + 360) % 360)

def create_folders(folder_names):
    for directory in folder_names:
        if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)

def get_reward_comp(vehicle, waypoint, collision):
    """
    Calculate reward components for the vehicle based on its position and orientation.
    
    Args:
        vehicle: The vehicle actor
        waypoint: The target waypoint
        collision: Collision status (1 if collision occurred, 0 otherwise)
    
    Returns:
        tuple: (cos_yaw_diff, distance, collision, speed) - reward components
    """
    # Get vehicle transform
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    
    # Calculate distance to waypoint
    distance = vehicle_location.distance(waypoint.transform.location)
    
    # Calculate yaw difference (orientation difference)
    vehicle_yaw = vehicle_rotation.yaw
    waypoint_yaw = waypoint.transform.rotation.yaw
    
    # Normalize yaw difference to [-180, 180]
    yaw_diff = waypoint_yaw - vehicle_yaw
    while yaw_diff > 180:
        yaw_diff -= 360
    while yaw_diff < -180:
        yaw_diff += 360
    
    # Convert to cosine similarity (1 = same direction, -1 = opposite direction)
    cos_yaw_diff = math.cos(math.radians(yaw_diff))
    
    # Get current speed
    speed = get_speed(vehicle)
    
    return cos_yaw_diff, distance, collision, speed

def reward_value(cos_yaw_diff, distance, collision, speed, target_speed):
    """
    Calculate the total reward based on reward components.
    
    Args:
        cos_yaw_diff: Cosine of yaw difference (orientation alignment)
        distance: Distance to target waypoint
        collision: Collision status (1 if collision occurred, 0 otherwise)
        speed: Current speed in km/h
        target_speed: Target speed in km/h
    
    Returns:
        float: Total reward value
    """
    # If collision, return a very large negative reward immediately
    if collision:
        return -500.0  # Even higher penalty for real-life safety
    
    # 1. Path Following Reward (Optimal Lane Centering)
    # We want to be exactly in the center of the lane (distance = 0)
    # Using a Gaussian-like reward for distance to center
    path_reward = 10.0 * math.exp(-(distance**2) / 0.5) 
    
    # 2. Orientation Reward (Smoothness)
    # High reward for being perfectly aligned with the road direction
    orientation_reward = 5.0 * cos_yaw_diff
    
    # 3. Speed Maintenance (Real-world efficiency)
    # Reward for maintaining target speed while aligned
    speed_reward = 0.0
    if cos_yaw_diff > 0.9: # Only reward speed if very well aligned
        speed_diff = abs(target_speed - speed)
        speed_reward = 5.0 * (1.0 - (speed_diff / target_speed))
    
    # 4. Smoothness Penalty (Real-life comfort)
    # Large steering angles are penalized to encourage smooth path following
    # Note: steering input would need to be passed here for a more direct penalty
    
    # Living reward:
    # Small positive reward for each step survived
    living_reward = 1.0
    
    # Combine all components
    total_reward = living_reward + path_reward + orientation_reward + speed_reward
    
    return total_reward