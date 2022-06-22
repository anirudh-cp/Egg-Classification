# Egg A mean value: (0, 0, 0.7), mean proportional_size = 2
# Egg A std dev value (0, 0, 0.15), std dev prop size = 0.36

# Egg B mean value: (0, 0, 0.45), mean proportional_size = 1.3
# Egg B std dev value (0, 0, 0.125), std dev prop size = 0.38

import numpy as np

import bpy
import bmesh
from math import radians

import datetime


params = {
    'A': {
        'count': 125,
        'proportioanlEditingMeanValueZ': 0.7,
        'proportionalEditingMeanSize': 2,
        'proportioanlEditingStdDevValueZ': 0.15,
        'proportionalEditingStdDevSize': 0.36,
        'ColorMean': (255, 249, 249),
        'ColorStdDev': (254, 235, 235),

        'CamAngleX': (0, 0),
        'CamAngleZ': (0, 0),
        'CamZ': (3.5, 5),

        'BackgroundColor': ((0, 0, 0), (150, 150, 150)),
        'LightSourceColor': ((255, 255, 255), (255, 247, 150)),
        'LightIntensity': (1000, 1000),
        'LightSourceZ': (5, 6),
        
        'Xmin': -2.1,
        'Xmax': 2.1,
        'Zmin': -0.15,
        'Zmax': 0.7,

        'PATH': 'D:\Anirudh\VIT\Second Year - Winter Sem\_ML Project\Images'
    },

    'B': {
        'count': 125,
        'proportioanlEditingMeanValueZ': 0.45,
        'proportionalEditingMeanSize': 1.3,
        'proportioanlEditingStdDevValueZ': 0.125,
        'proportionalEditingStdDevSize': 0.38,
        'ColorMean': (255, 189, 189),
        'ColorStdDev': (255, 173, 173),

        'CamAngleX': (0, 0),
        'CamAngleZ': (0, 0),
        'CamZ': (3.5, 5),

        'BackgroundColor': ((0, 0, 0), (150, 150, 150)),
        'LightSourceColor': ((255, 255, 255), (255, 247, 150)),
        'LightIntensity': (1000, 1000),
        'LightSourceZ': (5, 6),
        
        'Xmin': -2.1,
        'Xmax': 2.1,
        'Zmin': -0.15,
        'Zmax': 0.7,

        'PATH': 'D:\Anirudh\VIT\Second Year - Winter Sem\_ML Project\Images'
    },

}


def generateGradient(start, end, num):
    res = np.zeros((num, 3), dtype=np.float32)
    for i in range(3):
        res[:, i] = np.linspace(start[i], end[i], num)
    return res


def findClosestColor(color, colorList):
    def ErrorSquared(x, y):
        return (x-y)**2
    minDist = float('inf')
    minIndex = 0
    for index, item in enumerate(colorList):
        error = 0
        for i in range(3):
            error += ErrorSquared(color[i], item[i])
        if error < minDist:
            minDist = error
            minIndex = index
    return minIndex


def TruncatedNormal(mean, stdDev, min, max):
    value = np.random.normal(mean, stdDev)
    if value >= max:
        value = max
    if value <= min:
        value = min
    return value


def ValidateColor(color):
    res = []
    for item in color:
        res.append(int(item))
    return res


def randomizeValues(params, gradientObject, gradientLight):
    values = {}

    # Generate proportional editing value for Z using normal distribution
    values['proportionalEditingValueZ'] = np.random.normal(params['proportioanlEditingMeanValueZ'],
                                                           params['proportioanlEditingStdDevValueZ'])

    # Generate proportional editing size using normal distribution
    values['proportionalEditingSize'] = np.random.normal(params['proportionalEditingMeanSize'],
                                                         params['proportionalEditingStdDevSize'])

    # Generate color of object. Find the index of the closest color in the gradient set for the
    # mean and standard deviation colors. Use the indexes for normal distribution mean and standard
    # deviation. Find color at the index.
    values['Color'] = ValidateColor(gradientObject[int(
        TruncatedNormal(findClosestColor(params['ColorMean'], list(gradientObject)),
                        findClosestColor(params['ColorStdDev'], list(gradientObject)),
                        0, len(gradientObject)-1))
    ])

    # Generate camera orientation parameters using uniform distribution.
    values['CamAngleX'] = np.random.uniform(
        params['CamAngleX'][0], params['CamAngleX'][1])
    values['CamAngleZ'] = np.random.uniform(
        params['CamAngleZ'][0], params['CamAngleZ'][1])
    # values['CamZ'] = np.random.uniform(params['CamZ'][0], params['CamZ'][1])

    # Generate a random background color
    values['BackgroundColor'] = list(np.random.choice(range(
    params["BackgroundColor"][1][0]), size=3))

    # Generate a light source color from gradient set uniformly.
    values['LightSourceColor'] = ValidateColor(gradientLight[int(np.random.uniform(0, len(gradientLight)))])
    # Generate intensity of light source uniformly.
    values['LightIntensity'] = np.random.uniform(
        params['LightIntensity'][0], params['LightIntensity'][0])
    # Generate height at which light source is placed uniformly.
    values['LightSourceZ'] = np.random.uniform(
        params['LightSourceZ'][0], params['LightSourceZ'][1])
        
    values['X'] = np.random.uniform(params['Xmin'], params['Xmax'])
    values['Z'] = np.random.uniform(params['Zmin'], params['Zmax'])

    return values


def createRenderImageDebug(randomValues, path, file):
    for key, value in randomValues.items():
        print(key, value, file=file)
    print(path, file=file)


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
    texture = [x/255 for x in randomValues["Color"]]
    # print(texture)
    texture.append(1)   # For alpha channel
    bpy.data.materials["Egg"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = texture
    
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
    
    # Move the egg on the 2D plane of Y=0
    egg.location[0] = randomValues['X']
    egg.location[2] = randomValues['Z']

    # Render image.
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still = 1)

    # Delete sphere for next time.
    egg.select_set(True)
    bpy.ops.object.delete()


def main():
    # White to reddish pink gradient
    gradientObject = generateGradient((255, 255, 255), (255, 155, 155), 500)

    file = open(r'D:\Anirudh\VIT\Second Year - Winter Sem\_ML Project\ImageRenderLogs.txt', 'a+')
    print(f'\n\n{datetime.datetime.now()}\n' + '-'*16, file=file)

    count = 0    
    for variant in params:
        print(variant, file=file)
        gradientLight = generateGradient(params[variant]['LightSourceColor'][0],
                                         params[variant]['LightSourceColor'][1], 500)

        for index in range(params[variant]['count']):
            count += 1
            randomValues = randomizeValues(
                params[variant], gradientObject, gradientLight)
            createRenderImageDebug(randomValues, f"{params[variant]['PATH']}\image{count}", file)
            createRenderImage(randomValues,  f"{params[variant]['PATH']}\image{count}")
    
    print(f'\n{datetime.datetime.now()}\n' + '='*16, file=file)
    file.close()


if __name__ == '__main__':
    main()
