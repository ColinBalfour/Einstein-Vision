import bpy
import json
import os
import math

print("STARTING")

# ------------------------------------------------------------------------------
# User Configuration
# ------------------------------------------------------------------------------
json_folder = os.path.expanduser("~/Documents/RBE549/Einstein-Vision/json_outputs")
vehicle_blend_path = os.path.expanduser("~/Documents/RBE549/Einstein-Vision/P3Data/Assets/Vehicles/SedanAndHatchback.blend")
object_name = "Car"  # This is the object listed in OBJECTS in your .blend

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

# ------------------------------------------------------------------------------
# MAIN SCRIPT LOGIC
# ------------------------------------------------------------------------------
def import_vehicles_from_json():
    if not os.path.isdir(json_folder):
        print(f"[ERROR] JSON folder does not exist: {json_folder}")
        return

    for file_name in os.listdir(json_folder):
        if file_name.lower().endswith(".json"):
            json_path = os.path.join(json_folder, file_name)
            with open(json_path, "r") as f:
                data = json.load(f)

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

                print(f"[OK] Placed 'Vehicle' from {file_name} at ({x:.2f}, {y:.2f}, {z:.2f}), "
                      f"rotation=({roll:.2f}, {pitch:.2f}, {yaw:.2f})")

    print("[DONE] Finished importing all JSON Vehicles.")

# ------------------------------------------------------------------------------
# RUN THE FUNCTION
# ------------------------------------------------------------------------------
import_vehicles_from_json()

# ------------------------------------------------------------------------------
# OPTIONAL: AUTOMATIC RENDER
# ------------------------------------------------------------------------------
output_render_path = os.path.expanduser("~/Documents/RBE549/Einstein-Vision/render_result.png")
bpy.context.scene.render.filepath = output_render_path
bpy.ops.render.render(write_still=True)
print(f"[INFO] Rendered final scene to: {output_render_path}")
