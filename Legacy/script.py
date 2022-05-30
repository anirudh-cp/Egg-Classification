import bpy
import bmesh
from math import radians


def createRenderImage(randomValues, path):
    # Create sphere
    # bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32)

    # Create an empty mesh and the object.
    mesh = bpy.data.meshes.new('Sphere')
    basic_sphere = bpy.data.objects.new("Sphere", mesh)

    # Add the object into the scene.
    bpy.context.collection.objects.link(basic_sphere)

    # Select the newly created object
    bpy.context.view_layer.objects.active = basic_sphere
    basic_sphere.select_set(True)

    # Construct the bmesh sphere and assign it to the blender mesh.
    bm = bmesh.new()
    # Even though the parameter is diameter it is actually radius
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, diameter=1)
    bm.to_mesh(mesh)
    bm.free()

    # Select egg sphere
    egg = bpy.data.objects["Sphere"]
    egg.select_set(True)
    
    # Set to edit mode and deselect all vertices
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='DESELECT')

    # Get mesh of vertices
    egg_mesh = bmesh.from_edit_mesh(bpy.context.object.data)

    # Find and select the top vertex
    for v in egg_mesh.verts:
        if round(v.co.x, 4) == 0.0 and round(v.co.y, 4) == 0.0 and round(v.co.z, 4) == 1:
            v.select = True

    # Use proportional editing to make shape
    kwargs = {"orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(True, True, True), "mirror":True, "use_proportional_edit":True, "proportional_edit_falloff":'SMOOTH', "use_proportional_connected":True, "use_proportional_projected":False, "release_confirm":True}
    kwargs["value"] = (0, 0 ,randomValues["proportionalEditingValueZ"])
    kwargs["proportional_size"] = randomValues["proportionalEditingSize"]
    bpy.ops.transform.translate(**kwargs)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()
    
    # Set Camera angles
    camera = bpy.data.objects["Camera"]
    camera.rotation_euler[2] = radians(randomValues["CamAngleZ"])
    camera.rotation_euler[0] = radians(randomValues["CamAngleX"])
    
    # Set egg color
    texture =  bpy.data.materials["Egg"].node_tree.nodes["Principled BSDF"].inputs[0].default_value
    texture = [x/255 for x in randomValues["Color"]]
    print(texture)
    texture.append(1)   # For alpha channel
    
    mat = bpy.data.materials.get("Egg")
    if egg.data.materials:
        # assign to 1st material slot
        egg.data.materials[0] = mat
    else:
        # no slots
        egg.data.materials.append(mat)

    # Set background color
    color = [x/255 for x in randomValues["BackgroundColor"]]
    color.append(1)
    bpy.data.scenes["Scene"].node_tree.nodes["Alpha Over"].inputs[1].default_value = color
    
    # Set light source color
    bpy.data.lights["Light"].color = [x/255 for x in randomValues["LightSourceColor"]]
    color = [x/255 for x in randomValues["LightSourceColor"]]
    color.append(1)
    bpy.data.worlds["World"].node_tree.nodes["Principled BSDF"].inputs[17].default_value = color

    # Light source energy
    bpy.data.lights["Light"].energy = randomValues["LightIntensity"]
    
    # Set height of light source
    bpy.data.objects["Light"].location[2] = randomValues["LightSourceZ"]

    # Render image.
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still = 1)

    # Delete sphere for next time.
    egg.select_set(True)
    bpy.ops.object.delete()

'''

randomValues = {
    "proportionalEditingValueZ": 0.8499783990914019,
    "proportionalEditingSize": 2.220050368691664,
    "Color": [252, 223, 223],
    "CamAngleX": 9,
    "CamAngleZ": 360.8691150779494,
    "BackgroundColor": [50, 141, 142],
    "LightSourceColor": [255, 209, 94],
    "LightIntensity": 1000.0,
    "LightSourceZ": 5.0
}

createRenderImage(randomValues, 
"D:\Anirudh\VIT\Second Year - Winter Sem\_ML Project\A\image0", None)

'''