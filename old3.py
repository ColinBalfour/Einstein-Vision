
import bpy
import math

from blender_utils_setup import _ensure_emission, get_dimensions

# ─────────────────────────────────────────────────────────────────────────────
# THREE‑DISC TRAFFIC‑LIGHT RIG
# ─────────────────────────────────────────────────────────────────────────────

def _mk_disc(parent, tag, offsets, light_dir='left'):
    """Parent a thin cylinder that acts as one lamp disc."""
    x_offset, y_offset, z_offset, flip = offsets
    flip = -1 if flip else 1
    
    x_length, y_length, z_length = get_dimensions(parent)
    
    z_offset = z_offset + z_length / 2
    y_offset = (y_offset - (y_length / 2 - .05)) * flip
    
    if light_dir == 'left':
        x_offset = (x_offset - (x_length / 2 - 0.5)) * flip
    elif light_dir == 'right':
        x_offset = (x_offset + (x_length / 2 - 0.5)) * flip
    
    
    print(f"[INFO PAY ATTENTION] parent: {parent.name} x: {x_length:.2f} y: {y_length:.2f} z: {z_length:.2f}")
    
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.05)
    disc              = bpy.context.object
    disc.name         = f"{parent.name}_{tag}"
    disc.parent       = parent
    disc.location     = (x_offset, y_offset, z_offset)   # tweak forward offset if needed
    disc.rotation_euler = (math.radians(90), math.radians(0), math.radians(180))  # face the street
    mat               = bpy.data.materials.new(f"{disc.name}_MAT")
    disc.data.materials.append(mat)
    _ensure_emission(mat, (0, 0, 0), 0.0)       # start dark
    return disc

def _ensure_bulbs(parent, offsets):
    """Make sure the parent traffic light has R/Y/G discs once – idempotent."""
    existing = {c.name.split('_')[-1].lower() for c in parent.children}
    
    if not {'red', 'yellow', 'green'} <= existing:
        _mk_disc(parent, "red",    offsets, 'left')
        _mk_disc(parent, "yellow", offsets, 'right')

def set_brake_light(vehicle_obj, brake_on, offsets=None):
    """
    Illuminate exactly one of the discs on *traffic_obj*.
    Valid colours: 'red'|'yellow'|'green'.
    """
    colour  = "red"
    mapping = {"red": (1,0,0), "yellow": (1,1,0), "green": (0,1,0)}
    if colour not in mapping:
        print(f"[WARN] Unknown traffic‑light colour '{colour}' – defaulting to red")
        colour = "red"
        
    if offsets is None:
        offsets = (0, 0, 0, False) # x, y, z, flip
        
    offsets[2] = offsets[2] + 0.05 # z_offset for the brake light

    _ensure_bulbs(vehicle_obj, offsets)
    for child in vehicle_obj.children:
        tag = child.name.split('_')[-1].lower()
        if tag in mapping:
            mat = child.data.materials[0]
            on  = (tag == colour)
            _ensure_emission(mat, mapping[tag], strength = 25 if on else 0.05)

def set_turn_signal(vehicle_obj, direction, offsets=None):
    
    pass
    
    