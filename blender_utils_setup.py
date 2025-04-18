import bpy
import math

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