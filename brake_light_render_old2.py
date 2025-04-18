import bpy
import math

from blender_utils_setup import _ensure_emission

# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE LIGHTS RENDER MODULE (using flat discs for reliable emission)
# ─────────────────────────────────────────────────────────────────────────────

def _mk_disc(parent, tag, x_frac, z_frac, flip=False):
    """Parent a flat disc that acts as one vehicle light.
    x_frac and z_frac specify fraction of parent's X and Z for placement.
    flip=True places discs at +Y (front); flip=False places at -Y (rear)."""
    dims = parent.dimensions
    # create flat circle (NGON) for reliable emission
    radius = dims.x * 0.05
    bpy.ops.mesh.primitive_circle_add(radius=radius, fill_type='NGON')
    disc = bpy.context.object
    disc.name = f"{parent.name}_{tag}"
    disc.parent = parent
    # compute offsets
    y_sign = 1 if flip else -1
    y_offs = y_sign * (dims.y / 2 + radius)
    x_offs = x_frac * dims.x
    z_offs = z_frac * dims.z
    disc.location = (x_offs, y_offs, z_offs)
    # orient normal to face outward along Y-axis
    rot_x = math.radians(-90 if flip else 90)
    disc.rotation_euler = (rot_x, 0, 0)
    # assign emissive material default off
    mat = bpy.data.materials.new(f"{disc.name}_MAT")
    disc.data.materials.append(mat)
    _ensure_emission(mat, (0, 0, 0), 0.1)
    return disc

 
def _ensure_bulbs(parent, flip=False):
    """Ensure brake and turn discs exist once – idempotent."""
    existing = {c.name.split('_')[-1] for c in parent.children}
    for tag, xf, zf in [('brake_left', -0.25, 0.25),
                        ('brake_right', 0.25, 0.25),
                        ('turn_left', -0.3, 0.35),
                        ('turn_right', 0.3, 0.35)]:
        if tag not in existing:
            _mk_disc(parent, tag, xf, zf, flip)


# def set_brake_light(vehicle, brake_on, flip=False):
#     """Toggle brake lights (bright red) on a vehicle object. flip=True uses front."""
#     _ensure_bulbs(vehicle, flip)
#     on_strength = 100.0
#     off_strength = 0.1
#     for child in vehicle.children:
#         tag = child.name.split('_')[-1]
#         if 'brake' in tag:
#             mat = child.data.materials[0]
#             _ensure_emission(mat, (1, 0, 0), on_strength if brake_on else off_strength)


# def set_turn_signal(vehicle, direction, flip=False):
#     """Set turn signals (amber): direction ∈ {'left','right','hazard',None}. flip=True uses front."""
#     _ensure_bulbs(vehicle, flip)
#     on_strength = 80.0
#     off_strength = 0.1
#     for child in vehicle.children:
#         tag = child.name.split('_')[-1]
#         mat = child.data.materials[0]
#         if tag == 'turn_left':
#             on = direction in ('left', 'hazard')
#         elif tag == 'turn_right':
#             on = direction in ('right', 'hazard')
#         else:
#             continue
#         _ensure_emission(mat, (1, 0.6, 0), on_strength if on else off_strength)


