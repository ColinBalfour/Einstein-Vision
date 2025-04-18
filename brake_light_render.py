import bpy
import math

import mathutils

from blender_utils_setup import _ensure_emission


# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE LIGHTS  (brake + turn signals)
# ─────────────────────────────────────────────────────────────────────────────


# ── find rear axis automatically --------------------------------------------
def _rear_axis_vec(obj):
    """
    Return a *unit* vector in local space that points out the REAR of *obj*
    (the bbox face furthest from the origin in world coordinates).
    """
    axes = [mathutils.Vector(v) for v in
            [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0)]]
    max_d, rear = -1, None
    for a in axes:
        corner_world = obj.matrix_world @ (a * obj.dimensions / 2)
        dist = (corner_world - obj.location).length
        if dist > max_d:
            max_d, rear = dist, a
    return rear.normalized()

# ── lamp creation ------------------------------------------------------------
def _mk_lamp(parent, tag, size, rear_offset_m, sideways_frac, height_frac,
             colour, strength=0.02):
    """
    Generic creator for strip (brake) or cylinder (indicator) lamps.
    • size            : (x,y,z) cube half‑extents OR cylinder radius/depth.
    • rear_offset_m   : distance *behind* bumper plane.
    • sideways_frac   : ±fraction of bbox half‑width (0 = centre, 1 = outermost).
    • height_frac     : fraction of bbox height (0 = bottom, 1 = roof).
    """
    rear_vec = _rear_axis_vec(parent)
    right_vec = mathutils.Vector((rear_vec.y, -rear_vec.x, 0)).normalized()
    up_vec    = mathutils.Vector((0,0,1))

    # choose primitive
    if tag == "brake":
        bpy.ops.mesh.primitive_cube_add(size=1)
        lamp = bpy.context.object
        lamp.scale = (size[0]/2, size[1]/2, size[2]/2)
    else:  # indicator
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=size[1])
        lamp = bpy.context.object
        lamp.rotation_euler = (math.radians(90), 0, 0)

    lamp.name   = f"{parent.name}_{tag}"
    lamp.parent = parent

    half_w = parent.dimensions.x/2
    lamp.location = (
        rear_vec * -(rear_offset_m) +                    # behind bumper
        right_vec * (sideways_frac * half_w) +           # left / right
        up_vec    * (height_frac  * parent.dimensions.z) # vertical
    )
    mat = bpy.data.materials.new(f"{lamp.name}_MAT")
    lamp.data.materials.append(mat)
    _ensure_emission(mat, colour, strength)
    return lamp

# ── idempotent attachment ----------------------------------------------------
def _attach_vehicle_lights(vehicle):
    tags = {c.name.split('_')[-1] for c in vehicle.children}
    if {"brake", "ts_left", "ts_right"} <= tags:
        return                                   # already exists

    # sizes
    strip = (vehicle.dimensions.x*0.8, 0.03, 0.08)   # w, d, h
    ind_r, ind_d = 0.10, 0.04                       # radius, depth

    # create three lamps
    _mk_lamp(vehicle, "brake", strip,
             rear_offset_m=0.04,
             sideways_frac=0.0,   # centred
             height_frac =0.18,
             colour=(1,0,0))

    _mk_lamp(vehicle, "ts_left",(ind_r, ind_d,0),
             rear_offset_m=0.04,
             sideways_frac=-0.35,
             height_frac =0.16,
             colour=(1,0.5,0))

    _mk_lamp(vehicle, "ts_right",(ind_r, ind_d,0),
             rear_offset_m=0.04,
             sideways_frac= 0.35,
             height_frac =0.16,
             colour=(1,0.5,0))

# ── public controls ----------------------------------------------------------
def set_brake_light(vehicle, on, off_strength=0.02):
    _attach_vehicle_lights(vehicle)
    for ch in vehicle.children:
        if ch.name.endswith("_brake"):
            _ensure_emission(ch.data.materials[0], (1,0,0),
                             25 if on else off_strength)

def set_turn_signal(vehicle, direction, off_strength=0.02):
    """
    direction ∈ {'left','right','hazard', None}
    """
    _attach_vehicle_lights(vehicle)
    dir_low = None if direction is None else direction.lower()
    for ch in vehicle.children:
        if ch.name.endswith("_ts_left"):
            on = dir_low in {"left","hazard"}
            _ensure_emission(ch.data.materials[0], (1,0.5,0),
                             20 if on else off_strength)
        elif ch.name.endswith("_ts_right"):
            on = dir_low in {"right","hazard"}
            _ensure_emission(ch.data.materials[0], (1,0.5,0),
                             20 if on else off_strength)