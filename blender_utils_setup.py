import bpy
import math
from mathutils import Vector

def _ensure_emission(mat, rgb, strength):
    """Replace all nodes with a single emission of colour *rgb*."""
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out   = nodes.new("ShaderNodeOutputMaterial")
    emiss = nodes.new("ShaderNodeEmission")
    emiss.inputs["Color"].default_value     = (*rgb, 1.0)
    emiss.inputs["Strength"].default_value  = strength
    links.new(emiss.outputs["Emission"], out.inputs["Surface"])

def get_dimensions(obj):
    """
    Returns the (x, y, z) dimensions of an object in world space,
    accounting for scale, rotation, and position.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    
    world_coords = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
    
    xs = [v.x for v in world_coords]
    ys = [v.y for v in world_coords]
    zs = [v.z for v in world_coords]

    return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)