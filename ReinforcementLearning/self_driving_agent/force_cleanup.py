import glob
import os
import sys

# Add CARLA to path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def force_cleanup():
    """
    Forcefully resets the CARLA world to a clean state.
    Use this if the simulator crashes or becomes unresponsive between training sessions.
    """
    try:
        # Connect to client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get world
        world = client.get_world()
        
        # Disable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        print("Synchronous mode disabled.")
        
        # Tick to apply
        world.tick()
        
        # Destroy all actors
        print("Cleaning up actors...")
        actors = world.get_actors()
        
        # Destroy everything that can be destroyed
        for actor in actors:
            if any(x in actor.type_id for x in ['vehicle', 'sensor', 'walker', 'controller']):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except:
                    pass
        
        # Reload the world - this is the ONLY way to guarantee a 'like new' state
        print("Reloading Town02_Opt...")
        client.load_world('Town02_Opt')
        
        print("Force cleanup completed successfully! Simulator is now in a fresh state.")
        
    except Exception as e:
        print(f"Force cleanup failed: {e}")

if __name__ == "__main__":
    force_cleanup()
