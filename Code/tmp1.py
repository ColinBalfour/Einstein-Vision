import bpy
import json
import os
import math

print("STARTING")

# ------------------------------------------------------------------------------
# User Configuration
# ------------------------------------------------------------------------------
# Change this to the FOLDER containing JSON files
json_folder = os.path.expanduser("C:/Users/simra/Einstein-Vision/json_outputs/")
vehicle_blend_path = os.path.expanduser("C:/Users/simra/OneDrive - Worcester Polytechnic Institute (wpi.edu)/2. Computer Vision\P3Data/P3Data/Assets/Vehicles/SedanAndHatchback.blend")
object_name = "Car"  # This is the object listed in OBJECTS in your .blend
car_scale_factor = 4.5 / 62.45261764526367  

# ------------------------------------------------------------------------------
# HELPER FUNCTION: Append an Object
# ------------------------------------------------------------------------------
def append_object(blend_file, obj_name):
    """
    Appends an object named obj_name from blend_file and returns the new object.
    """
    existing_objs = set(bpy.data.objects)

    directory = blend_file + "/Object/"
    filepath  = directory + obj_name

    print(f"[DEBUG] Attempting to append object '{obj_name}' from {filepath}")
    bpy.ops.wm.append(
        filepath=filepath,
        directory=directory,
        filename=obj_name,
        link=False
    )

    new_objs = set(bpy.data.objects) - existing_objs
    if not new_objs:
        print(f"[ERROR] No new object appended for '{obj_name}' from {blend_file}")
        return None

    new_obj = new_objs.pop()
    print(f"[DEBUG] Appended object: {new_obj.name}")
    return new_obj

# Add this function to your script
def setup_camera(location=(0, 0, 0), rotation=(0, 0, 0), target_obj=None):
    """
    Sets up the active camera with the given location and rotation,
    or points it at a target object if specified.
    
    Parameters:
    - location: (x, y, z) tuple for camera position
    - rotation: (roll, pitch, yaw) tuple in radians
    - target_obj: Optional object to point the camera at
    """
    # Get the active camera or create one if none exists
    if bpy.context.scene.camera:
        camera = bpy.context.scene.camera
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
    
    # Set camera location
    camera.location = location
    
    # Either set direct rotation or point at target
    if target_obj:
        # Point camera at target object
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        # Apply constraint temporarily to set rotation
        bpy.context.view_layer.update()
        # If you want to keep the rotation but remove the constraint:
        # camera.rotation_euler = camera.rotation_euler.copy()
        # camera.constraints.remove(constraint)
    else:
        # Set direct rotation
        camera.rotation_euler = rotation
    
    return camera

# ------------------------------------------------------------------------------
# MAIN SCRIPT LOGIC
# ------------------------------------------------------------------------------
def import_vehicles_from_json():
    # Check if it's a single JSON file
    if os.path.isfile(json_folder):
        json_files = [json_folder]
    # Or a directory with multiple JSON files
    elif os.path.isdir(json_folder):
        json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) 
                    if f.lower().endswith(".json")]
    else:
        print(f"[ERROR] JSON path does not exist: {json_folder}")
        return
    
                
    
    # Process each JSON file
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            file_name = os.path.basename(json_path)
            
            # Check if this is a vehicle object
            if data.get("name") == "Vehicle":
                pose = data["object_data"].get("pose")
                if not pose or len(pose) < 3:
                    print(f"[WARN] No valid pose in {json_path}")
                    continue

                # Append the "Car" object from Vehicle.blend
                appended_obj = append_object(vehicle_blend_path, object_name)
                if not appended_obj:
                    continue

                # Unpack the pose (xyz camera -> xzy blender)
                appended_obj.scale = (car_scale_factor, car_scale_factor, car_scale_factor)
                bpy.context.view_layer.objects.active = appended_obj
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                print(f"[INFO] Car dimensions after scaling: {appended_obj.dimensions}")
                
               
                scale_factor = 10
                x, y, z = pose[0] * scale_factor, -pose[2] * scale_factor, pose[1] * scale_factor
                roll = pitch = yaw = 0
                if len(pose) >= 6:
                    roll  = math.radians(pose[3])
                    pitch = math.radians(pose[4])
                    yaw   = math.radians(pose[5])

                # Set object transform
                appended_obj.location = (x, y, z)
                appended_obj.rotation_euler = (roll, pitch, yaw)
                
                # Make sure object is visible in viewport and render
                appended_obj.hide_viewport = False
                appended_obj.hide_render = False

                # Add debugging info
                print(f"[OK] Placed 'Vehicle' from {file_name} at ({x:.2f}, {y:.2f}, {z:.2f}), "
                    f"rotation=({roll:.2f}, {pitch:.2f}, {yaw:.2f})")
                
                # Focus view on this object
                bpy.ops.object.select_all(action='DESELECT')
                appended_obj.select_set(True)
                bpy.context.view_layer.objects.active = appended_obj
                bpy.ops.view3d.view_selected(use_all_regions=False)
        
        except Exception as e:
            print(f"[ERROR] Failed to process {json_path}: {str(e)}")

    print("[DONE] Finished importing all JSON Vehicles.")

# ------------------------------------------------------------------------------
# RUN THE FUNCTION
# ------------------------------------------------------------------------------
camera_position = (5, -100, 25)  # X, Y, Z position
camera_rotation = (-math.radians(10), math.radians(45), math.radians(90))  # Roll, Pitch, Yaw
camera = setup_camera(location=camera_position, rotation=camera_rotation)
import_vehicles_from_json()

# ------------------------------------------------------------------------------
# OPTIONAL: AUTOMATIC RENDER
# ------------------------------------------------------------------------------
output_render_path = os.path.expanduser("C:/Users/simra/Einstein-Vision/")
bpy.context.scene.render.filepath = output_render_path
bpy.ops.render.render(write_still=True)
print(f"[INFO] Rendered final scene to: {output_render_path}")