import bpy
import math

from blender_utils_setup import _ensure_emission

# ─────────────────────────────────────────────────────────────────────────────
# THREE‑DISC TRAFFIC‑LIGHT RIG
# ─────────────────────────────────────────────────────────────────────────────

def _mk_disc(parent, tag, z_offset):
    """Parent a thin cylinder that acts as one lamp disc."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.05)
    disc              = bpy.context.object
    disc.name         = f"{parent.name}_{tag}"
    disc.parent       = parent
    disc.location     = (0.2, z_offset, -.05)   # tweak forward offset if needed
    disc.rotation_euler = (math.radians(0), math.radians(90), math.radians(0))  # face the street
    mat               = bpy.data.materials.new(f"{disc.name}_MAT")
    disc.data.materials.append(mat)
    _ensure_emission(mat, (0, 0, 0), 0.0)       # start dark
    return disc

def _ensure_bulbs(parent):
    """Make sure the parent traffic light has R/Y/G discs once – idempotent."""
    existing = {c.name.split('_')[-1].lower() for c in parent.children}
    if not {'red', 'yellow', 'green'} <= existing:
        _mk_disc(parent, "red",    0.2)
        _mk_disc(parent, "yellow", 0.65)
        _mk_disc(parent, "green",  1.1)

def highlight_light_color(traffic_obj, colour):
    """
    Illuminate exactly one of the discs on *traffic_obj*.
    Valid colours: 'red'|'yellow'|'green'.
    """
    colour  = colour.lower()
    mapping = {"red": (1,0,0), "yellow": (1,1,0), "green": (0,1,0)}
    if colour not in mapping:
        print(f"[WARN] Unknown traffic‑light colour '{colour}' – defaulting to red")
        colour = "red"

    _ensure_bulbs(traffic_obj)
    for child in traffic_obj.children:
        tag = child.name.split('_')[-1].lower()
        if tag in mapping:
            mat = child.data.materials[0]
            on  = (tag == colour)
            _ensure_emission(mat, mapping[tag], strength = 25 if on else 0.05)

def add_direction_arrow(traffic_obj, direction):
    """
    Parent a simple emissive arrow mesh to the traffic light.
    • direction ∈ {'left','right','straight'}
    """
    direction = direction.lower()
    if direction not in {"left", "right", "straight"}:
        return

    # crude arrow: a cone + cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.05)
    body = bpy.context.object
    body.location = (0.0, 0.12, 0.0)   # adjust to lens position
    _ensure_emission(body.data.materials.new("ArrowBody"), (1, 1, 1), 15)

    bpy.ops.mesh.primitive_cone_add(radius1=0.08, depth=0.06)
    head = bpy.context.object
    head.location = (0.0, 0.12, 0.03)
    _ensure_emission(head.data.materials.new("ArrowHead"), (1, 1, 1), 15)

    # orient
    rot_z = {"left":  math.radians( 90),
             "right": math.radians(-90),
             "straight": 0}[direction]
    for obj in (body, head):
        obj.rotation_euler = (math.radians(90), 0, rot_z)
        obj.parent = traffic_obj