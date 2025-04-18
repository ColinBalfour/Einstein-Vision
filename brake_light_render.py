import bpy, math
from blender_utils_setup import _ensure_emission, get_dimensions

# ─────────────────────────────────────────────────────────────────────────────
# Colour definitions (RGB, 0–1 range)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_MAP = {
    'red':    (1.0, 0.0, 0.0),
    'yellow': (1.0, 0.65, 0.0),   # amber
}

def _mk_disc(parent, tag, offsets, light_dir):
    """
    Make one lamp disc of colour `tag` on side `light_dir` ('left' or 'right').
    Naming: parent_lightdir_tag so we can parse both.
    """
    x_off, y_off, z_off, flip = offsets
    flip = -1 if flip else 1
    x_len, y_len, z_len = get_dimensions(parent)

    # center on parent
    z_off = z_off + z_len/2
    y_off = (y_off - (y_len/2 - .05)) * flip
    if light_dir == 'left':
        x_off = -(x_off + (x_len/2 - .5)) * flip
    else:
        x_off = (x_off + (x_len/2 - .5)) * flip

    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.05)
    disc = bpy.context.object
    disc.name = f"{parent.name}_{light_dir}_{tag}"
    disc.parent = parent
    disc.location = (x_off, y_off, z_off)
    disc.rotation_euler = (math.radians(90), 0, math.radians(180))

    mat = bpy.data.materials.new(f"{disc.name}_MAT")
    disc.data.materials.append(mat)
    # start “off” with a low tint so you see the lens
    _ensure_emission(mat, COLOR_MAP[tag], strength=0.05)
    return disc

def set_brake_light(vehicle, brake_on, offsets=None):
    """
    Ensures two red discs exist (left & right), then sets them on or off.
    """
    tag = 'red'
    offsets = offsets or [0,0,0,False]
    offsets = (offsets[0] - .05, offsets[1], offsets[2] - 0.1, offsets[3]) # slightly lower for brake

    # 1) ensure we have exactly two red bulbs
    existing = {c.name for c in vehicle.children}
    for side in ('left','right'):
        name = f"{vehicle.name}_{side}_{tag}"
        if name not in existing:
            _mk_disc(vehicle, tag, offsets, side)

    # 2) switch them on/off
    for child in vehicle.children:
        parts = child.name.split('_')
        if parts[-1] == tag:
            mat = child.data.materials[0]
            strength = 25.0 if brake_on else 0.1
            _ensure_emission(mat, COLOR_MAP[tag], strength)

def set_turn_signal(vehicle, direction, offsets=None):
    """
    Ensures two yellow discs exist (left & right), then flashes the one matching
    `direction` ('left' or 'right').
    """
    tag = 'yellow'
    offsets = offsets or [0,0,0,False]
    offsets = (offsets[0] + .05, offsets[1], offsets[2] + 0.1, offsets[3])  # slightly higher for signal

    # 1) ensure we have exactly two yellow bulbs
    existing = {c.name for c in vehicle.children}
    for side in ('left','right'):
        name = f"{vehicle.name}_{side}_{tag}"
        if name not in existing:
            _mk_disc(vehicle, tag, offsets, side)

    # 2) only the matching side glows bright
    for child in vehicle.children:
        parts = child.name.split('_')
        if parts[-1] == tag:
            mat = child.data.materials[0]
            on = (parts[-2] == direction) or (direction == 'hazard')
            strength = 25.0 if on else 0.2
            _ensure_emission(mat, COLOR_MAP[tag], strength)