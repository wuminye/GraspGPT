import bpy
import bmesh
import mathutils
from mathutils import Vector
import random
import os
import glob

def clear_scene():
    """Clear all mesh objects in the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

def import_ply_file(filepath):
    """Import PLY file and return the created object"""
    bpy.ops.import_mesh.ply(filepath=filepath)
    return bpy.context.active_object

def create_bbox_walls(bbox_min, bbox_max):
    """Create bbox boundary walls as collision boundaries"""
    walls = []
    
    # Floor
    bpy.ops.mesh.primitive_cube_add(size=2)
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = ((bbox_max[0] - bbox_min[0])/2, (bbox_max[1] - bbox_min[1])/2, 0.2)
    floor.location = ((bbox_max[0] + bbox_min[0])/2, (bbox_max[1] + bbox_min[1])/2, bbox_min[2] - 0.2)
    
    # Set as rigid body
    bpy.context.view_layer.objects.active = floor
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    floor.rigid_body.type = 'PASSIVE'

    walls.append(floor)
    
    # Create four walls
    wall_thickness = 0.02
    wall_height = (bbox_max[2] - bbox_min[2]) + 1
    
    # Two walls in X direction
    for x_pos, name in [(bbox_min[0] - wall_thickness/2, "Wall_X_Min"), 
                        (bbox_max[0] + wall_thickness/2, "Wall_X_Max")]:
        bpy.ops.mesh.primitive_cube_add(size=2)
        wall = bpy.context.active_object
        wall.name = name
        wall.scale = (wall_thickness/2, (bbox_max[1] - bbox_min[1])/2 + wall_thickness, wall_height/2)
        wall.location = (x_pos, (bbox_max[1] + bbox_min[1])/2, (bbox_max[2] + bbox_min[2])/2)
        
        bpy.context.view_layer.objects.active = wall
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        wall.rigid_body.type = 'PASSIVE'
        wall.hide_render = True
        wall.hide_viewport = True
        walls.append(wall)
    
    # Two walls in Y direction
    for y_pos, name in [(bbox_min[1] - wall_thickness/2, "Wall_Y_Min"), 
                        (bbox_max[1] + wall_thickness/2, "Wall_Y_Max")]:
        bpy.ops.mesh.primitive_cube_add(size=2)
        wall = bpy.context.active_object
        wall.name = name
        wall.scale = ((bbox_max[0] - bbox_min[0])/2 + wall_thickness, wall_thickness/2, wall_height/2)
        wall.location = ((bbox_max[0] + bbox_min[0])/2, y_pos, (bbox_max[2] + bbox_min[2])/2)
        
        bpy.context.view_layer.objects.active = wall
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        wall.rigid_body.type = 'PASSIVE'
        wall.hide_render = True
        wall.hide_viewport = True
        walls.append(wall)
    
    return walls

def setup_physics_world():
    """Setup physics world"""
    scene = bpy.context.scene
    
    # Enable rigid body world
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    
    # Set gravity - handle different Blender versions
    rbw = scene.rigidbody_world
    if hasattr(rbw, 'gravity'):
        rbw.gravity = (0, 0, -9.81)
    elif hasattr(rbw, 'effector_weights'):
        rbw.effector_weights.gravity = 9.81
    
    # Set time scale
    if hasattr(rbw, 'time_scale'):
        rbw.time_scale = 1.0
    
    # Improve solver settings to reduce jittering
    if hasattr(rbw, 'solver_iterations'):
        rbw.solver_iterations = 20  # Increase solver iterations
    if hasattr(rbw, 'steps_per_second'):
        rbw.steps_per_second = 120  # Increase simulation steps

def add_physics_to_object(obj, mass=1.0):
    """Add rigid body physics to object"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    obj.rigid_body.type = 'ACTIVE'
    obj.rigid_body.mass = mass
    obj.rigid_body.collision_shape = 'CONVEX_HULL'
    obj.rigid_body.friction = 0.8  # Increase friction to reduce sliding
    obj.rigid_body.restitution = 0.1  # Reduce bounce to prevent jittering
    
    # Add damping to reduce oscillations
    obj.rigid_body.linear_damping = 0.1  # Linear damping
    obj.rigid_body.angular_damping = 0.1  # Angular damping
    
    # Use better collision margins
    obj.rigid_body.collision_margin = 0.001  # Smaller collision margin
    
    # Use mesh collision shape for better accuracy if needed
    # Uncomment the next line for more accurate but slower collision detection
    # obj.rigid_body.collision_shape = 'MESH'

def get_random_position_in_bbox(bbox_min, bbox_max, drop_height_offset=0.5):
    """Generate random position within bbox range, dropping from above"""
    x = random.uniform(bbox_min[0] + 0.05, bbox_max[0] - 0.05)  # Leave boundary margin
    y = random.uniform(bbox_min[1] + 0.05, bbox_max[1] - 0.05)  # Leave boundary margin
    z = bbox_max[2] + drop_height_offset  # Drop from above bbox top
    return Vector((x, y, z))

def remove_physics_from_objects(objects):
    """Remove physics properties from objects to preserve final geometry"""
    for obj in objects:
        if obj and obj.rigid_body:
            # Select the object
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Remove rigid body physics
            bpy.ops.rigidbody.object_remove()
            
            # Apply all transforms to make the current position permanent
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            
            obj.select_set(False)

def simulate_physics_drop(ply_files_folder, bbox_min, bbox_max, num_objects=10, simulation_frames=250):
    """
    Main function: Load PLY files and perform physics simulation
    
    Parameters:
    - ply_files_folder: Path to folder containing PLY files
    - bbox_min: bbox minimum coordinates (x, y, z)
    - bbox_max: bbox maximum coordinates (x, y, z)  
    - num_objects: Number of objects to place
    - simulation_frames: Number of frames for physics simulation
    """
    
    # Clear scene
    clear_scene()
    
    # Get all PLY files
    #ply_files = glob.glob(os.path.join(ply_files_folder, "*.ply"))
    ply_files = [os.path.join(ply_files_folder, '%03d'%f, 'nontextured_simplified_blender_3.ply') for f in range(88)]
    if not ply_files:
        print("No PLY files found!")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    
    # Setup physics world
    setup_physics_world()
    
    # Create bbox boundaries
    walls = create_bbox_walls(Vector(bbox_min), Vector(bbox_max))
    print("Created bbox boundary walls")
    
    # Randomly select and import PLY files
    selected_files = random.sample(ply_files, min(num_objects, len(ply_files)))
    imported_objects = []
    
    for i, ply_file in enumerate(selected_files):
        print(f"Importing object {i+1}: {os.path.basename(ply_file)}")
        
        # Import PLY file
        obj = import_ply_file(ply_file)
        if obj:
            # Set random position
            drop_position = get_random_position_in_bbox(Vector(bbox_min), Vector(bbox_max))
            obj.location = drop_position
            
            # Random rotation
            obj.rotation_euler = (
                random.uniform(0, 6.28),  # 0 to 2π
                random.uniform(0, 6.28),
                random.uniform(0, 6.28)
            )
            
            # Add physics properties
            add_physics_to_object(obj, mass=random.uniform(0.5, 2.0))
            imported_objects.append(obj)
            
            # Slight delay after importing each object to avoid simultaneous dropping
            bpy.context.scene.frame_set(i * 10)
    
    # Run physics simulation
    print(f"Starting physics simulation for {simulation_frames} frames...")
    
    # Set timeline
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = simulation_frames
    bpy.context.scene.frame_set(1)
    
    # Run simulation
    for frame in range(1, simulation_frames + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
    
    # After simulation, remove physics to preserve final geometry
    print("Simulation complete! Removing physics properties to preserve final geometry...")
    #remove_physics_from_objects(imported_objects)
    #remove_physics_from_objects(walls)
    
    # Reset timeline to show final result
    bpy.context.scene.frame_set(simulation_frames)
    
    print("Physics simulation completed!")
    print(f"Successfully imported and simulated {len(imported_objects)} objects")
    print("Final geometry preserved - objects will maintain their settled positions")

# Example usage
if __name__ == "__main__":
    # Configuration parameters
    PLY_FILES_FOLDER = "H:\\code\\GraspGPT\\data\\models"  # Replace with your PLY files path
    BBOX_MIN = (-0.27, -0.18, 0)    # bbox minimum coordinates
    BBOX_MAX = (0.27, 0.18, 0.2)      # bbox maximum coordinates
    NUM_OBJECTS = 3          # Number of objects to place
    SIMULATION_FRAMES = 60   # Physics simulation frames
    
    # Execute simulation
    simulate_physics_drop(
        ply_files_folder=PLY_FILES_FOLDER,
        bbox_min=BBOX_MIN,
        bbox_max=BBOX_MAX,
        num_objects=NUM_OBJECTS,
        simulation_frames=SIMULATION_FRAMES
    )