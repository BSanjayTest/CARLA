import glob
import os
import sys

# Import carla from the CARLA egg file
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# dqn action values
action_values = [-0.75, -0.5, -0.25, -0.15, -0.1, -0.05, 0,
                0.05, 0.1, 0.15, 0.25, 0.5, 0.75]
action_map = {i:x for i, x in enumerate(action_values)}

# Available weather presets for training scenarios
# These are the standard CARLA weather presets
WEATHER_PRESETS = {
    'ClearNoon': carla.WeatherParameters.ClearNoon,
    'CloudyNoon': carla.WeatherParameters.CloudyNoon,
    'WetNoon': carla.WeatherParameters.WetNoon,
    'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
    'ClearSunset': carla.WeatherParameters.ClearSunset,
    'CloudySunset': carla.WeatherParameters.CloudySunset,
    'WetSunset': carla.WeatherParameters.WetSunset,
    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
    'ClearNight': carla.WeatherParameters.ClearNight,
    'CloudyNight': carla.WeatherParameters.CloudyNight,
    'WetNight': carla.WeatherParameters.WetNight,
    'WetCloudyNight': carla.WeatherParameters.WetCloudyNight,
    'SoftRainNight': carla.WeatherParameters.SoftRainNight,
    'HardRainNight': carla.WeatherParameters.HardRainNight,
}

# Default scenario to train - change this to train different scenarios
# Options: 'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 
#          'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset',
#          'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight',
#          'SoftRainNight', 'HardRainNight', 'ClearMidnight'
TRAINING_SCENARIO = 'ClearNoon'  # Change this to your desired scenario

# Or train on multiple scenarios (list of scenarios)
TRAINING_SCENARIOS = ['ClearNoon', 'CloudyNoon', 'ClearNight', 'CloudyNight']

env_params = {
    'target_speed' :30 , 
    'max_iter': 1000,  # Reduced from 4000 for faster episodes
    'start_buffer': 5,  # Reduced from 10
    'train_freq': 1,
    'save_freq': 50,   # Save more frequently to see progress
    'start_ep': 0,
    'max_dist_from_waypoint': 20,
    'visuals': True,
    
    # Enhanced environment settings
    'num_other_vehicles': 15,  # Increased NPC vehicles
    'num_pedestrians': 20,     # Added pedestrians
    'weather_preset': TRAINING_SCENARIO,  # Use specific weather preset
    'vehicle_types': [  # Available vehicle types for variety
        'vehicle.nissan.patrol',
        'vehicle.tesla.model3', 
        'vehicle.ford.mustang',
        'vehicle.audi.tt',
        'vehicle.nissan.micra'
    ]
}