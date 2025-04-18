# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE LAMPS v3  (robust rear detection)
# Places lamps flush to the true REAR bumper plane, no matter which
# axis each vehicle mesh uses for “forward”.
# ─────────────────────────────────────────────────────────────────────────────
import bpy, math, mathutils
import bmesh

# ═══════════════════════════════════════════════════════════════════════════ #
# 1)  Universal emissive‑material helper
# ═══════════════════════════════════════════════════════════════════════════ #
def _ensure_emission(mat, rgb, strength):
    if mat.users > 1:                      # somebody linked this material? clone it
        mat = mat.copy()
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value    = (*rgb, 1.0)
    emis.inputs["Strength"].default_value = strength
    links.new(emis.outputs["Emission"], out.inputs["Surface"])


# ═══════════════════════════════════════════════════════════════════════════ #
# 2)  Tag each vehicle with its forward axis (once, after append+scale)
# ═══════════════════════════════════════════════════════════════════════════ #
def set_vehicle_axes(vehicle, nose_axis="Y+"):
    """
    Record which *local* axis points FORWARD for this vehicle mesh.
    Allowed strings: 'X+','X-','Y+','Y-'
    """
    vehicle["_nose_axis"] = nose_axis.upper()

# ═══════════════════════════════════════════════════════════════════════════ #
# 3)  Compute rear / side / up world‑vectors and rear‑plane distance
# ═══════════════════════════════════════════════════════════════════════════ #
def _basis_and_rear_offset(obj, rear_margin=0.04):
    """
    Returns:
      rear_vec   – world unit‑vector out of rear bumper
      side_vec   – world +right unit‑vector (passenger side, LHD)
      up_vec     – world +Z unit‑vector
      rear_dist  – distance from obj origin to REAR bumper plane + margin
    The distance is positive so ` rear_vec * rear_dist ` lands *behind* bumper.
    """
    axis_map = {
        'X+': mathutils.Vector(( 1, 0, 0)),
        'X-': mathutils.Vector((-1, 0, 0)),
        'Y+': mathutils.Vector(( 0, 1, 0)),
        'Y-': mathutils.Vector(( 0,-1, 0)),
    }
    nose_local = axis_map.get(obj.get("_nose_axis", "Y+"))
    R = obj.matrix_world.to_3x3().normalized()

    fwd_vec   = (R @ nose_local).normalized()
    rear_vec  = -fwd_vec
    up_vec    = R @ mathutils.Vector((0,0,1))
    side_vec  = rear_vec.cross(up_vec).normalized()

    # bounding‑box half‑length along forward axis (object space)
    if nose_local.x:     half_len = obj.dimensions.x / 2
    else:                half_len = obj.dimensions.y / 2

    rear_dist = half_len + rear_margin
    return rear_vec, side_vec, up_vec.normalized(), rear_dist

# ═══════════════════════════════════════════════════════════════════ #
# 4)  Create one lamp (cube strip or cylinder indicator) and parent it
#    — world‑space position first, then parent —
# ═══════════════════════════════════════════════════════════════════ #
def _mk_lamp(parent, tag,
             dims, rear_vec, side_vec, up_vec, rear_dist,
             side_offset_m, height_frac,
             colour, strength=0.02):

    # 1. create mesh -------------------------------------------------
    if tag == "brake":
        bpy.ops.mesh.primitive_cube_add(size=1)
        lamp = bpy.context.object
        lamp.scale = (dims[0]/2, dims[1]/2, dims[2]/2)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=dims[0], depth=dims[1])
        lamp = bpy.context.object

    lamp.name = f"{parent.name}_{tag}"
    lamp.rotation_mode = 'QUATERNION'
    lamp.rotation_quaternion = (rear_vec).to_track_quat('Z', 'Y')

    # 2. world‑space position  (extra ½‑depth outwards) --------------
    depth_offset = dims[1] / 2         # push it fully clear of the body
    world_pos = (rear_vec * (rear_dist + depth_offset) +
                 side_vec * side_offset_m +
                 up_vec   * (height_frac * parent.dimensions.z))
    lamp.location = world_pos

    # 3. parent & material ------------------------------------------
    lamp.parent = parent                                   # keeps world pose
    mat = bpy.data.materials.new(name=f"{lamp.name}_MAT")
    lamp.data.materials.clear(); lamp.data.materials.append(mat)
    _ensure_emission(mat, colour, strength)
    lamp.visible_camera = True
    return lamp


# ═══════════════════════════════════════════════════════════════════════════ #
# 5)  Attach three lamps once per vehicle (idempotent)
# ═══════════════════════════════════════════════════════════════════════════ #
def _attach_vehicle_lights(vehicle):
    tags = {c.name.split('_')[-1] for c in vehicle.children}
    if {"brake","ts_left","ts_right"} <= tags:
        return

    rear_vec, side_vec, up_vec, rear_dist = _basis_and_rear_offset(vehicle)

    # sizes
    strip_w = vehicle.dimensions.x * 0.8
    strip   = (strip_w, 0.03, 0.08)          # width, depth, height
    ind_r, ind_d = 0.10, 0.05                # radius, depth

    half_w = vehicle.dimensions.x / 2

    _mk_lamp(vehicle, "brake", strip,
             rear_vec, side_vec, up_vec, rear_dist,
             side_offset_m = 0.0,
             height_frac   = 0.50,      # <‑‑ centred vertically
             colour        = (1,0,0))

    _mk_lamp(vehicle, "ts_left",  (ind_r, ind_d),
             rear_vec, side_vec, up_vec, rear_dist,
             side_offset_m = -(half_w*0.35),
             height_frac   = 0.50,      # <‑‑ centred vertically
             colour        = (1,0.5,0))

    _mk_lamp(vehicle, "ts_right", (ind_r, ind_d),
             rear_vec, side_vec, up_vec, rear_dist,
             side_offset_m =  (half_w*0.35),
             height_frac   = 0.50,      # <‑‑ centred vertically
             colour        = (1,0.5,0))


# ═══════════════════════════════════════════════════════════════════════════ #
# 6)  Public toggles
# ═══════════════════════════════════════════════════════════════════════════ #
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
    d = None if direction is None else direction.lower()
    for ch in vehicle.children:
        if ch.name.endswith("_ts_left"):
            on = d in {"left","hazard"}
            _ensure_emission(ch.data.materials[0], (1,0.5,0),
                             20 if on else off_strength)
        elif ch.name.endswith("_ts_right"):
            on = d in {"right","hazard"}
            _ensure_emission(ch.data.materials[0], (1,0.5,0),
                             20 if on else off_strength)
