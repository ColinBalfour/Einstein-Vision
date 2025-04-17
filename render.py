import bpy
import json
import os
import sys
import math

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
from traffic_light_render import highlight_light_color, add_direction_arrow

# Enable GPU rendering in Cycles
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'  # or 'OPTIX', 'HIP', 'METAL'

# Force Blender to detect devices
prefs.get_devices()
prefs.refresh_devices()

# Print and enable devices
for device in prefs.devices:
    device.use = True
    print(f"[INFO] Device: {device.name}, Type: {device.type}, Enabled: {device.use}")

# Set the scene to use GPU rendering
bpy.context.scene.cycles.device = 'GPU'
print("[INFO] GPU rendering is now set to use:", bpy.context.scene.cycles.device)

print("STARTING")

# ------------------------------------------------------------------------------
# User Configuration
# ------------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
# base_path = os.path.expanduser("~/Documents/RBE549/Einstein-Vision")
json_folder = os.path.join(base_path, "json_outputs")

object_assets = {
    'car': os.path.join(base_path, "P3Data/Assets/Vehicles/SedanAndHatchback.blend"),
    'truck': os.path.join(base_path, "P3Data/Assets/Vehicles/Truck.blend"),
    'SUV': os.path.join(base_path, "P3Data/Assets/Vehicles/SUV.blend"),
    'bicycle': os.path.join(base_path, "P3Data/Assets/Vehicles/Bicycle.blend"),
    'pickup': os.path.join(base_path, "P3Data/Assets/Vehicles/PickupTruck.blend"),
    'motorcycle': os.path.join(base_path, "P3Data/Assets/Vehicles/Motorcycle.blend"),
    'TrafficLight': os.path.join(base_path, "P3Data/Assets/TrafficSignal.blend"),
    'Pedestrian': os.path.join(base_path, "P3Data/Assets/Pedestrain.blend"),
    'RoadSign': os.path.join(base_path, "P3Data/Assets/StopSign.blend"),
}

object_names = {
    'car': "Car",
    'truck': "Truck",
    'SUV': "Jeep_3_",
    'bicycle': "roadbike 2.0.1",
    'TrafficLight': "Traffic_signal1",
    'pickup': "PickupTruck",
    'motorcycle': "Bike", # B_wheel
    'Pedestrian': "BaseMesh_Man_Simple",
    'RoadSign': "StopSign_Geo",
}

object_scales = {
    'car': 4 / 62.45261764526367,
    'truck': 6 / 8.25,
    'SUV': 1,
    'bicycle': 1.5 / 3.91,
    'pickup': 5 / 9.47,
    'motorcycle': 2.5 / 8.78,
    'TrafficLight': .4,
    'Pedestrian': 1.75 / 6.3432,
    'RoadSign': 2.1336 / 6.7066,
}

object_rotations = {
    'car': (0, 0, 180),
    'truck': (0, 0, 0),
    'SUV': (0, 0, 0),
    'bicycle': (90, 0, 0),
    'pickup': (90, 0, 90),
    'motorcycle': (90, 0, 90),
    'TrafficLight': (90, 0, -90),
    'Pedestrian': (90, 0, 0),
    'RoadSign': (90, 0, -90),
}

object_offsets = {
    'car': (0, 0, 0),
    'truck': (0, 0, 0),
    'SUV': (0, 0, 0),
    'bicycle': (0, 0, 0),
    'pickup': (0, 0, 1.4),
    'motorcycle': (0, 0, 0),
    'TrafficLight': (0, 0, -0),
    'Pedestrian': (0, 0, 0),
    'RoadSign': (0, 0, -0),
}

# ------------------------------------------------------------------------------
# Setup scene to match GUI and CLI
# ------------------------------------------------------------------------------
def setup_scene():
    if 'Scene' in bpy.data.scenes:
        bpy.context.window.scene = bpy.data.scenes['Scene']
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 64

    # Add light if missing
    has_light = any(obj.type == 'LIGHT' for obj in bpy.data.objects)
    if not has_light:
        light_data = bpy.data.lights.new(name="MainLight", type='SUN')
        light_obj = bpy.data.objects.new(name="MainLight", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (5, -5, 10)
        light_data.energy = 5

    for obj in bpy.data.objects:
        obj.hide_render = False
        obj.hide_viewport = False
        for mod in obj.modifiers:
            mod.show_render = True

    print("[SETUP] Scene setup complete.")

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def append_object(blend_file, obj_name):
    existing_objs = set(bpy.data.objects)
    directory = blend_file + "/Object/"
    
    if obj_name == "Bike":
        directory = blend_file + "/Collection/"
    
    filepath = directory + obj_name
    print(f"[DEBUG] Appending object '{obj_name}' from {filepath}")
    bpy.ops.wm.append(filepath=filepath, directory=directory, filename=obj_name, link=False)
    new_objs = set(bpy.data.objects) - existing_objs
    return new_objs.pop() if new_objs else None

def assign_lane_color(obj, color_rgba):
    mat = bpy.data.materials.new(name="LaneMat")
    mat.diffuse_color = color_rgba
    mat.use_nodes = False
    obj.data.materials.append(mat)

def create_lane_curve(world_points, lane_name="Lane"):
    curve_data = bpy.data.curves.new(name=lane_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 64
    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(world_points) - 1)
    for i, (x, y, z) in enumerate(world_points):
        spline.points[i].co = (x, z, -y, 1)
    lane_obj = bpy.data.objects.new(lane_name, curve_data)
    bpy.context.scene.collection.objects.link(lane_obj)
    curve_data.bevel_depth = 0.05
    return lane_obj

def setup_camera(location=(0, 0, 0), rotation=(0, 0, 0), target_obj=None):
    if bpy.context.scene.camera:
        camera = bpy.context.scene.camera
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
    camera.location = location
    if target_obj:
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        bpy.context.view_layer.update()
    else:
        camera.rotation_euler = rotation
    return camera

# ------------------------------------------------------------------------------
# Import JSON Objects
# ------------------------------------------------------------------------------
def import_objects_from_json():
    if not os.path.isdir(json_folder):
        print(f"[ERROR] JSON folder does not exist: {json_folder}")
        return
    for file_name in os.listdir(json_folder):
        if file_name.lower().endswith(".json"):
            json_path = os.path.join(json_folder, file_name)
            with open(json_path, "r") as f:
                data = json.load(f)
            if 'object_data' not in data:
                continue
            pose = data["object_data"].get("pose")
            if not pose or len(pose) < 3:
                print(f"[WARN] No valid pose in {json_path}")
                continue
            
            asset_name = data['name']
            if asset_name == "Vehicle":
                asset_name = data.get("vehicle_type", "car")
                
            obj_asset = object_assets[asset_name]
            obj_name = object_names[asset_name]
            
            appended_obj = append_object(obj_asset, obj_name)
            if not appended_obj:
                continue
            
            bpy.context.view_layer.objects.active = appended_obj
            scale_factor = object_scales[asset_name]
            bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            x, y, z = pose[0], pose[2], -pose[1]
            rot = object_rotations[asset_name]
            roll = math.radians(rot[0])
            pitch = math.radians(rot[1])
            yaw = math.radians(rot[2])
            if len(pose) >= 6:
                roll += math.radians(pose[3])
                pitch += math.radians(pose[4])
                yaw += math.radians(pose[5])
                
            x_off, y_off, z_off = object_offsets[asset_name]    
            appended_obj.location = (x + x_off, y + y_off, z + z_off)    
            
            appended_obj.rotation_euler = (roll, pitch, yaw)
            appended_obj.hide_viewport = False
            appended_obj.hide_render = False
            
            # ─── traffic‑light specific tweaks ────────────────────────────────
            if asset_name == "TrafficLight":
                highlight_light_color(
                    appended_obj,
                    data.get("color", "red")       # default red
                )
                arrow_dir = data.get("arrow")      # may be None
                if arrow_dir and arrow_dir.lower() is not None:
                    add_direction_arrow(appended_obj, arrow_dir)
            # ──────────────────────────────────────────────────────────────────
            
            
            print(f"[OK] Placed '{asset_name}' from {file_name} at ({x:.2f}, {y:.2f}, {z:.2f})")

def import_lanes_from_json():
    if not os.path.isdir(json_folder):
        print(f"[ERROR] Lane JSON folder does not exist: {json_folder}")
        return
    for file_name in os.listdir(json_folder):
        if file_name.lower().endswith(".json"):
            json_path = os.path.join(json_folder, file_name)
            with open(json_path, "r") as f:
                data = json.load(f)
            if data.get("name") == "Lane":
                lane_type = data.get("lane_type", "unknown-type")
                world_coords = data.get("world_coords")
                if not world_coords:
                    continue
                lane_obj = create_lane_curve(world_coords, lane_name=f"Lane_{lane_type}")
                if lane_type == "divider-line":
                    assign_lane_color(lane_obj, (1, 1, 0, 1))
                lane_obj.location = (0, 0, 0)
                lane_obj.rotation_euler = (0, 0, 0)
                print(f"[OK] Created lane '{lane_type}' from {file_name}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
setup_scene()

camera_position = (-.5, -.5, 1.5)
camera_rotation = (math.radians(90 - 10), math.radians(0), math.radians(0))
camera = setup_camera(location=camera_position, rotation=camera_rotation)

import_objects_from_json()
import_lanes_from_json()

output_render_path = os.path.join(base_path, "outputs/render_result.png")
bpy.context.scene.render.filepath = output_render_path
bpy.ops.render.render(write_still=True)
print(f"[INFO] Rendered final scene to: {output_render_path}")
