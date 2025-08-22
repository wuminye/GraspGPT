import bpy
import bmesh
import mathutils
from mathutils import Vector
import random
import os
import glob

def clear_scene():
    """Clear all mesh objects in the scene"""
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')
    
    # Delete all selected objects
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    
    # Alternative method: delete objects directly
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Clear all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

def import_ply_file(filepath):
    """Import PLY file and return the created object"""
    bpy.ops.import_mesh.ply(filepath=filepath)
    return bpy.context.active_object

def preload_all_objects(ply_files_folder):
    """Preload all 88 objects into memory and return dictionary of meshes"""
    print("Preloading all 88 objects into memory...")
    preloaded_meshes = {}
    
    for model_id in range(88):
        ply_file = os.path.join(ply_files_folder, f'{model_id:03d}', 'nontextured_simplified_blender_3.ply')
        if os.path.exists(ply_file):
            # Import object temporarily
            obj = import_ply_file(ply_file)
            if obj and obj.data:
                # Store mesh data
                mesh_copy = obj.data.copy()
                mesh_copy.name = f"mesh_{model_id:03d}"
                preloaded_meshes[model_id] = mesh_copy
                print(f"Loaded object {model_id:03d}")
                
                # Remove the temporary object
                bpy.data.objects.remove(obj, do_unlink=True)
    
    print(f"Successfully preloaded {len(preloaded_meshes)} objects")
    return preloaded_meshes

def create_object_from_mesh(mesh_data, model_id):
    """Create a new object from preloaded mesh data"""
    # Create object from mesh
    obj = bpy.data.objects.new(f"object_{model_id:03d}", mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj

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

def set_object_color_by_id(obj, model_id):
    """Set material color based on model ID"""
    # Generate color based on model ID
    color = (model_id/255.0, model_id/255.0, model_id/255.0, 1.0)
    
    # Create or get material
    material_name = f"Material_{model_id:03d}"
   
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color

    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

def save_final_meshes(output_dir, imported_objects, object_ids, scene_id=0):
    """Save final frame meshes to target directory, excluding walls and floors"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clear selection
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select all objects except walls and floors
    valid_objects = []
    for i, obj in enumerate(imported_objects):
        if obj and obj.name and not obj.name.startswith(('Wall_', 'Floor')):
            # Set object color based on model ID
            model_id = object_ids[i] if i < len(object_ids) else i
            set_object_color_by_id(obj, model_id)
            
            # Select the object
            obj.select_set(True)
            valid_objects.append(obj)
    
    if valid_objects:
        # Clear all selections first to ensure proper material display
        bpy.ops.object.select_all(action='DESELECT')
        
        # Re-select valid objects
        for obj in valid_objects:
            obj.select_set(True)
        
        # Set one of the objects as active
        bpy.context.view_layer.objects.active = valid_objects[0]
        
        # Export OBJ format only with scene ID
        obj_path = os.path.join(output_dir, f"scene_{scene_id:04d}.obj")
        
        # Export OBJ with materials
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=True,
            use_mesh_modifiers=False,
            use_normals=False,
            use_uvs=False,
            use_materials=True,
            use_vertex_groups=False,
            global_scale=1.0
        )
        
        print(f"Saved {len(valid_objects)} objects to {output_dir}")
    
    # Clear selection
    bpy.ops.object.select_all(action='DESELECT')

def simulate_physics_drop_with_preloaded(preloaded_meshes, bbox_min, bbox_max, selected_objects, simulation_frames=250, output_dir=None, scene_id=0):
    """
    Main function: Perform physics simulation with preloaded objects
    
    Parameters:
    - preloaded_meshes: Dictionary of preloaded mesh data
    - bbox_min: bbox minimum coordinates (x, y, z)
    - bbox_max: bbox maximum coordinates (x, y, z)  
    - selected_objects: List of model IDs to use in this scene
    - simulation_frames: Number of frames for physics simulation
    - output_dir: Directory to save final meshes (optional)
    - scene_id: Scene identifier for naming
    """
    
    print(f"Generating scene {scene_id} with {len(selected_objects)} objects")
    
    # Setup physics world
    setup_physics_world()
    
    # Create bbox boundaries
    walls = create_bbox_walls(Vector(bbox_min), Vector(bbox_max))
    print("Created bbox boundary walls")
    
    # Create objects from preloaded meshes
    imported_objects = []
    object_ids = selected_objects.copy()
    
    for i, model_id in enumerate(selected_objects):
        print(f"Creating object {i+1}: model {model_id:03d}")
        
        # Create object from preloaded mesh
        if model_id in preloaded_meshes:
            obj = create_object_from_mesh(preloaded_meshes[model_id], model_id)
            
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
    
    # Save final meshes if output directory is specified
    if output_dir:
        print(f"Saving final meshes to {output_dir}...")
        save_final_meshes(output_dir, imported_objects, object_ids, scene_id)
    
    print("Physics simulation completed!")
    print(f"Successfully imported and simulated {len(imported_objects)} objects")
    print("Final geometry preserved - objects will maintain their settled positions")
    
    # Clear scene for next simulation
    clear_scene()

def generate_multiple_scenes(ply_files_folder, bbox_min, bbox_max, num_scenes=10, simulation_frames=60, output_dir=None):
    """Generate multiple random scenes with varying object counts"""
    print(f"Starting generation of {num_scenes} scenes...")
    
    # Preload all objects once
    preloaded_meshes = preload_all_objects(ply_files_folder)
    available_objects = list(preloaded_meshes.keys())
    
    if not available_objects:
        print("No objects were loaded!")
        return
    
    # Generate each scene
    for scene_id in range(num_scenes):
        # Random number of objects (3-6)
        num_objects = random.randint(3, 6)
        
        # Randomly select objects for this scene
        selected_objects = random.sample(available_objects, min(num_objects, len(available_objects)))
        
        print(f"\n--- Scene {scene_id + 1}/{num_scenes} ---")
        print(f"Objects: {selected_objects}")
        
        # Run simulation for this scene
        simulate_physics_drop_with_preloaded(
            preloaded_meshes=preloaded_meshes,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            selected_objects=selected_objects,
            simulation_frames=simulation_frames,
            output_dir=output_dir,
            scene_id=scene_id
        )
    
    print(f"\nCompleted generation of {num_scenes} scenes!")
    
    # Clean up preloaded meshes
    for mesh in preloaded_meshes.values():
        bpy.data.meshes.remove(mesh, do_unlink=True)

# Example usage
if __name__ == "__main__":
    # Configuration parameters
    PLY_FILES_FOLDER = "H:\\code\\GraspGPT\\data\\models"  # Replace with your PLY files path
    BBOX_MIN = (-0.27, -0.18, 0)    # bbox minimum coordinates
    BBOX_MAX = (0.27, 0.18, 0.2)      # bbox maximum coordinates
    NUM_SCENES = 2000          # Number of scenes to generate
    SIMULATION_FRAMES = 60   # Physics simulation frames
    OUTPUT_DIR = "H:\\code\\GraspGPT\\output\\synthetic_meshes"  # Directory to save final meshes
    
    # Execute multiple scene generation
    generate_multiple_scenes(
        ply_files_folder=PLY_FILES_FOLDER,
        bbox_min=BBOX_MIN,
        bbox_max=BBOX_MAX,
        num_scenes=NUM_SCENES,
        simulation_frames=SIMULATION_FRAMES,
        output_dir=OUTPUT_DIR
    )