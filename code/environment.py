from OpenGL.GL import *

import easy_shaders as es
import scene_graph as sg
import basic_shapes as bs
from operations import *

#Arreglo que contendra los objetos del ambiente
Environment_elements = []

#Clase element, que representa un objeto de ambiente o una textura que se desplaza con cierto movimiento
class Element:
    def __init__(self, node):
        self.time = 0 #parametro que cuantifica el movimiente en cada coordenada del espacio
        self.x_function = lambda t : t * 0 #funcion con la que se mueve en el eje x, dependiente de time
        self.y_function = lambda t : t * 0 #funcion con la que se mueve en el eje y, dependiente de time
        self.z_function = lambda t : t * 0 #funcion con la que se mueve en el eje z, dependiente de time
        self.node_to = node # Referencia al nodo en que se encuentra, se usa para aplicar las transformaciones
        self.scale = [1,1,1] # escala
        self.rotation = [0, 0, 0] #rotacion en los ejes X, Y, Z respectivamente

        self.exit_pos = [0, 0, 0] #posicion para revisar si se ha salido de l escena
        self.can_leave = False #booleano que indica si el objeto sale de escena o no

        self.can_move = True #booleano que indica si el objeto puede moverse
        self.time_to_spawn = 0 #parametro que indica el tiempo en que demora en aparecer en escena de nuevo
        self.time_counter = 0 #contador de tiempo

    #Metodo para actualizar el objeto
    def update(self, delta):
        if self.can_move:
            #si se puede mover, se actualiza el teimpo y la posicion segun sus funciones en cada componente
            self.time += delta
            pos_x = self.x_function(self.time)
            pos_y = self.y_function(self.time)
            pos_z = self.z_function(self.time)

            #se aplican las rotaciones
            rot = tr.identity()
            if self.rotation[0] != 0:
                rot = tr.matmul([tr.rotationX(self.rotation[0]), rot])
            if self.rotation[1] != 0:
                rot = tr.matmul([tr.rotationY(self.rotation[1]), rot])
            if self.rotation[2] != 0:
                rot = tr.matmul([tr.rotationZ(self.rotation[2]), rot])

            #se aplican las transformacion al nodo apuntado
            self.node_to.transform = tr.matmul([
                tr.translate(pos_x, pos_y, pos_z),
                rot,
                tr.scale(self.scale[0], self.scale[1], self.scale[2])
            ])

            #Si el objeto puede salirse de la escena y
            # si la posicion de salida no es 0 en su componente y lo excede, se sale de la escena
            if self.can_leave:
                if self.exit_pos[0] != 0:
                    if np.abs(pos_x) > np.abs(self.exit_pos[0]):
                        self.can_move = False
                if self.exit_pos[1] != 0:
                    if np.abs(pos_y) > np.abs(self.exit_pos[1]):
                        self.can_move = False
                if self.exit_pos[2] != 0:
                    if np.abs(pos_z) > np.abs(self.exit_pos[2]):
                        self.can_move = False
        else:
            if self.time_counter > self.time_to_spawn:
                self.can_move = True
                self.time = 0
                self.time_counter = 0
            else:
                self.time_counter += delta

#Funcion que crea la skybox, texturas para el fondo
def createSkyBoxNode(distance):
    r = distance/2
    box_node = sg.SceneGraphNode("sky_box")

    #NX
    vertices_0 = [
        -r, -r, -r, 0, 1, #BOTTOM - LEFT
        -r,  r, -r, 1, 1, #BOTTOM - RIGHT
        -r,  r,  r, 1, 0,   #TOP - RIGHT
        -r, -r,  r, 0, 0]   #TOP - LEFT
    indices_0 = [
        0, 1, 2,
        2, 3, 0]
    fileName_0 = "Sprites/nx.png"
    shape_0 = bs.Shape(vertices_0, indices_0, fileName_0)
    quad_node_0 = sg.SceneGraphNode("quad")
    quad_node_0.childs += [es.toGPUShape(shape_0, GL_CLAMP_TO_EDGE, GL_LINEAR)]
    box_node.childs += [quad_node_0]

    # NY
    vertices_1 = [
        -r, -r, -r, 0, 1,  # BOTTOM - LEFT
         r, -r, -r, 1, 1,  # BOTTOM - RIGHT
         r,  r, -r, 1, 0,  # TOP - RIGHT
        -r,  r, -r, 0, 0]  # TOP - LEFT
    indices_1 = [
        0, 1, 2,
        2, 3, 0]
    fileName_1 = "Sprites/ny.png"
    shape_1 = bs.Shape(vertices_1, indices_1, fileName_1)
    quad_node_1 = sg.SceneGraphNode("quad")
    quad_node_1.childs += [es.toGPUShape(shape_1, GL_CLAMP_TO_EDGE, GL_LINEAR)]
    box_node.childs += [quad_node_1]

    # NZ
    vertices_2 = [
         r, -r, -r, 0, 1,  # BOTTOM - LEFT
        -r, -r, -r, 1, 1,  # BOTTOM - RIGHT
        -r, -r,  r, 1, 0,  # TOP - RIGHT
         r, -r,  r, 0, 0]  # TOP - LEFT
    indices_2 = [
        0, 1, 2,
        2, 3, 0]
    fileName_2 = "Sprites/nz.png"
    shape_2 = bs.Shape(vertices_2, indices_2, fileName_2)
    quad_node_2 = sg.SceneGraphNode("quad")
    quad_node_2.childs += [es.toGPUShape(shape_2, GL_CLAMP_TO_EDGE, GL_LINEAR)]
    box_node.childs += [quad_node_2]

    # PX
    vertices_3 = [
         r,  r, -r, 0, 1,  # BOTTOM - LEFT
         r, -r, -r, 1, 1,  # BOTTOM - RIGHT
         r, -r,  r, 1, 0,  # TOP - RIGHT
         r,  r,  r, 0, 0]  # TOP - LEFT
    indices_3 = [
        0, 1, 2,
        2, 3, 0]
    fileName_3 = "Sprites/px.png"
    shape_3 = bs.Shape(vertices_3, indices_3, fileName_3)
    quad_node_3 = sg.SceneGraphNode("quad")
    quad_node_3.childs += [es.toGPUShape(shape_3, GL_CLAMP_TO_EDGE, GL_LINEAR)]
    box_node.childs += [quad_node_3]

    # PY
    vertices_4 = [
        -r,  r,  r, 0, 1,  # BOTTOM - LEFT
         r,  r,  r, 1, 1,  # BOTTOM - RIGHT
         r, -r,  r, 1, 0,  # TOP - RIGHT
        -r, -r,  r, 0, 0]  # TOP - LEFT
    indices_4 = [
        0, 1, 2,
        2, 3, 0]
    fileName_4 = "Sprites/py.png"
    shape_4 = bs.Shape(vertices_4, indices_4, fileName_4)
    quad_node_4 = sg.SceneGraphNode("quad")
    quad_node_4.childs += [es.toGPUShape(shape_4, GL_CLAMP_TO_EDGE, GL_LINEAR)]
    box_node.childs += [quad_node_4]

    # PZ
    vertices_5 = [
        -r,  r, -r, 0, 1,  # BOTTOM - LEFT
         r,  r, -r, 1, 1,  # BOTTOM - RIGHT
         r,  r,  r, 1, 0,  # TOP - RIGHT
        -r,  r,  r, 0, 0]  # TOP - LEFT
    indices_5 = [
        0, 1, 2,
        2, 3, 0]
    fileName_5 = "Sprites/pz.png"
    shape_5 = bs.Shape(vertices_5, indices_5, fileName_5)
    quad_node_5 = sg.SceneGraphNode("quad")
    quad_node_5.childs += [es.toGPUShape(shape_5, GL_CLAMP_TO_EDGE, GL_LINEAR)]
    box_node.childs += [quad_node_5]

    return box_node

#Funcion que inicializa los elementos
def SetupEnvironment(textured_node):
    global Environment_elements

    #Se crea la skybox y se anade al nodo correspondiente
    sky_box = createSkyBoxNode(80)
    textured_node.childs += [sky_box]

    #Se cargan y se guardan las texturas
    Shapes = [
        es.toGPUShape(bs.createTextureQuad("Sprites/cloud_0.png"), GL_REPEAT, GL_LINEAR),     # Cloud 0    - 0
        es.toGPUShape(bs.createTextureQuad("Sprites/cloud_1.png"), GL_REPEAT, GL_LINEAR),     # Cloud 1    - 1
        es.toGPUShape(bs.createTextureQuad("Sprites/cloud_2.png"), GL_REPEAT, GL_LINEAR),     # Cloud 2    - 2
        es.toGPUShape(bs.createTextureQuad("Sprites/cloud_3.png"), GL_REPEAT, GL_LINEAR),     # Cloud 3    - 3
        es.toGPUShape(bs.createTextureQuad("Sprites/cloud_4.png"), GL_REPEAT, GL_LINEAR),     # Cloud 4    - 4
        es.toGPUShape(bs.createTextureQuad("Sprites/cloud_5.png"), GL_REPEAT, GL_LINEAR),     # Cloud 5    - 5
        es.toGPUShape(bs.createTextureQuad("Sprites/balloon_0.png"), GL_REPEAT, GL_LINEAR),   # Balloon 0  - 6
        es.toGPUShape(bs.createTextureQuad("Sprites/balloon_1.png"), GL_REPEAT, GL_LINEAR),   # Balloon 1  - 7
        es.toGPUShape(bs.createTextureQuad("Sprites/zeppelli_0.png"), GL_REPEAT, GL_LINEAR),  # Zeppelin 0 - 8
        es.toGPUShape(bs.createTextureQuad("Sprites/zeppelli_1.png"), GL_REPEAT, GL_LINEAR),  # Zeppelin 1 - 9
        es.toGPUShape(bs.createTextureQuad("Sprites/dron.png"), GL_REPEAT, GL_LINEAR),        # Drone 0    - 10
        es.toGPUShape(bs.createTextureQuad("Sprites/plane.png"), GL_REPEAT, GL_LINEAR),       # Plane 0    - 11
        es.toGPUShape(bs.createTextureQuad("Sprites/jet.png"), GL_REPEAT, GL_LINEAR),         # Jet 0      - 12
        es.toGPUShape(bs.createTextureQuad("Sprites/great_fox.png"), GL_REPEAT, GL_LINEAR),   # Great Fox  - 13
        es.toGPUShape(bs.createTextureQuad("Sprites/falcon.png"), GL_REPEAT, GL_LINEAR)       # Falcon 0   - 14
    ]

    #Se crean los elementos de ambiente, se ajustan sus valores y funciones, y se anade al arreglo

    # FALCON
    falcon_node = sg.SceneGraphNode("great_fox")
    falcon_node.childs = [Shapes[14]]
    textured_node.childs += [falcon_node]
    falcon = Element(falcon_node)
    falcon.time = 0
    falcon.x_function = lambda t: 30 + 0 * t
    falcon.y_function = lambda t: -30 + 0 * t
    falcon.z_function = lambda t: -60 + 2 * t
    falcon.scale = [5, 35, 1]
    falcon.rotation = [np.pi / 2, 0, np.pi/8]
    falcon.can_leave = True
    falcon.exit_pos = [0, 0, 80]
    Environment_elements += [falcon]

    #ZEPPELIN 0
    zeppelin_1_node = sg.SceneGraphNode("zeppelin_1")
    zeppelin_1_node.childs = [Shapes[9]]
    textured_node.childs += [zeppelin_1_node]
    zeppelin_1 = Element(zeppelin_1_node)
    zeppelin_1.time = 0
    zeppelin_1.x_function = lambda t: -30 + 0 * t
    zeppelin_1.y_function = lambda t: 60 - 1.2 * t
    zeppelin_1.z_function = lambda t: 35 - 0.4 * t
    zeppelin_1.scale = [15, 7.5, 1]
    zeppelin_1.rotation = [np.pi/2, 0, -np.pi/2]
    zeppelin_1.can_leave = True
    zeppelin_1.exit_pos = [0, -60, 0]
    Environment_elements += [zeppelin_1]

    # ZEPPELIN 0
    zeppelin_0_node = sg.SceneGraphNode("zeppelin_0")
    zeppelin_0_node.childs = [Shapes[8]]
    textured_node.childs += [zeppelin_0_node]
    zeppelin_0 = Element(zeppelin_0_node)
    zeppelin_0.time = 0
    zeppelin_0.x_function = lambda t: 30 + 0 * t
    zeppelin_0.y_function = lambda t: -60 + 3 * t
    zeppelin_0.z_function = lambda t: 20 - 1 * t
    zeppelin_0.scale = [14, 7, 1]
    zeppelin_0.rotation = [np.pi / 2, 0, -np.pi / 2]
    zeppelin_0.can_leave = True
    zeppelin_0.exit_pos = [0, 60, 0]
    Environment_elements += [zeppelin_0]

    # BALLON 0
    ballon_0_node = sg.SceneGraphNode("ballon_0")
    ballon_0_node.childs = [Shapes[6]]
    textured_node.childs += [ballon_0_node]
    ballon_0 = Element(ballon_0_node)
    ballon_0.time = 0
    ballon_0.x_function = lambda t: -28 + 0 * t
    ballon_0.y_function = lambda t: -28 + 0 * t
    ballon_0.z_function = lambda t: 5 + 13 * np.sin(t* 0.1)
    ballon_0.scale = [10, 15, 1]
    ballon_0.rotation = [np.pi / 2, 0, -np.pi / 4]
    ballon_0.can_leave = False
    Environment_elements += [ballon_0]

    # BALLON 1
    ballon_1_node = sg.SceneGraphNode("ballon_1")
    ballon_1_node.childs = [Shapes[7]]
    textured_node.childs += [ballon_1_node]
    ballon_1 = Element(ballon_1_node)
    ballon_1.time = 0
    ballon_1.x_function = lambda t: 28 + 0 * t
    ballon_1.y_function = lambda t: 28 + 0 * t
    ballon_1.z_function = lambda t: 10 + 10 * np.sin(t * 0.2)
    ballon_1.scale = [10, 15, 1]
    ballon_1.rotation = [np.pi / 2, 0, -np.pi / 4]
    ballon_1.can_leave = False
    Environment_elements += [ballon_1]

    #LOWER CLOUDS
    cloud_0_node = sg.SceneGraphNode("cloud_0")
    cloud_0_node.childs = [Shapes[5]]
    textured_node.childs += [cloud_0_node]
    cloud_0 = Element(cloud_0_node)
    cloud_0.time = 0
    cloud_0.x_function = lambda t: -69 + 2.5 * t
    cloud_0.y_function = lambda t: 10 + 0 * t
    cloud_0.z_function = lambda t: -39 + 0 * t
    cloud_0.scale = [60, 40, 1]
    cloud_0.can_leave = True
    cloud_0.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_0]

    cloud_6_node = sg.SceneGraphNode("cloud_6")
    cloud_6_node.childs = [Shapes[3]]
    textured_node.childs += [cloud_6_node]
    cloud_6 = Element(cloud_6_node)
    cloud_6.time = 35
    cloud_6.x_function = lambda t: -69 + 2.5 * t
    cloud_6.y_function = lambda t: 10 + 0 * t
    cloud_6.z_function = lambda t: -38 + 0 * t
    cloud_6.scale = [50, 25, 1]
    cloud_6.can_leave = True
    cloud_6.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_6]

    cloud_1_node = sg.SceneGraphNode("cloud_1")
    cloud_1_node.childs = [Shapes[4]]
    textured_node.childs += [cloud_1_node]
    cloud_1 = Element(cloud_1_node)
    cloud_1.time = 0
    cloud_1.x_function = lambda t: -69 + 3.0 * t
    cloud_1.y_function = lambda t: -30 + 0 * t
    cloud_1.z_function = lambda t: -36 + 0 * t
    cloud_1.scale = [35, 22, 1]
    cloud_1.can_leave = True
    cloud_1.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_1]

    cloud_7_node = sg.SceneGraphNode("cloud_7")
    cloud_7_node.childs = [Shapes[0]]
    textured_node.childs += [cloud_7_node]
    cloud_7 = Element(cloud_7_node)
    cloud_7.time = 30
    cloud_7.x_function = lambda t: -69 + 3.0 * t
    cloud_7.y_function = lambda t: -30 + 0 * t
    cloud_7.z_function = lambda t: -35 + 0 * t
    cloud_7.scale = [42, 25, 1]
    cloud_7.can_leave = True
    cloud_7.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_7]

    cloud_2_node = sg.SceneGraphNode("cloud_2")
    cloud_2_node.childs = [Shapes[3]]
    textured_node.childs += [cloud_2_node]
    cloud_2 = Element(cloud_2_node)
    cloud_2.time = 0
    cloud_2.x_function = lambda t: -69 + 3.5 * t
    cloud_2.y_function = lambda t: 30 + 0 * t
    cloud_2.z_function = lambda t: -33 + 0 * t
    cloud_2.scale = [42, 31, 1]
    cloud_2.can_leave = True
    cloud_2.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_2]

    cloud_8_node = sg.SceneGraphNode("cloud_8")
    cloud_8_node.childs = [Shapes[2]]
    textured_node.childs += [cloud_8_node]
    cloud_8 = Element(cloud_8_node)
    cloud_8.time = 25
    cloud_8.x_function = lambda t: -69 + 3.5 * t
    cloud_8.y_function = lambda t: 30 + 0 * t
    cloud_8.z_function = lambda t: -32 + 0 * t
    cloud_8.scale = [46, 23, 1]
    cloud_8.can_leave = True
    cloud_8.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_8]

    cloud_3_node = sg.SceneGraphNode("cloud_3")
    cloud_3_node.childs = [Shapes[2]]
    textured_node.childs += [cloud_3_node]
    cloud_3 = Element(cloud_3_node)
    cloud_3.time = 0
    cloud_3.x_function = lambda t: -69 + 4.0 * t
    cloud_3.y_function = lambda t: -15 + 0 * t
    cloud_3.z_function = lambda t: -30 + 0 * t
    cloud_3.scale = [40, 30, 1]
    cloud_3.can_leave = True
    cloud_3.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_3]

    cloud_9_node = sg.SceneGraphNode("cloud_9")
    cloud_9_node.childs = [Shapes[5]]
    textured_node.childs += [cloud_9_node]
    cloud_9 = Element(cloud_9_node)
    cloud_9.time = 20
    cloud_9.x_function = lambda t: -69 + 4.0 * t
    cloud_9.y_function = lambda t: -15 + 0 * t
    cloud_9.z_function = lambda t: -29 + 0 * t
    cloud_9.scale = [23, 13, 1]
    cloud_9.can_leave = True
    cloud_9.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_9]

    cloud_4_node = sg.SceneGraphNode("cloud_4")
    cloud_4_node.childs = [Shapes[1]]
    textured_node.childs += [cloud_4_node]
    cloud_4 = Element(cloud_4_node)
    cloud_4.time = 0
    cloud_4.x_function = lambda t: -69 + 4.5 * t
    cloud_4.y_function = lambda t:  0 + 0 * t
    cloud_4.z_function = lambda t: -27 + 0 * t
    cloud_4.scale = [43, 32, 1]
    cloud_4.can_leave = True
    cloud_4.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_4]

    cloud_10_node = sg.SceneGraphNode("cloud_10")
    cloud_10_node.childs = [Shapes[4]]
    textured_node.childs += [cloud_10_node]
    cloud_10 = Element(cloud_10_node)
    cloud_10.time = 15
    cloud_10.x_function = lambda t: -69 + 4.5 * t
    cloud_10.y_function = lambda t: 0 + 0 * t
    cloud_10.z_function = lambda t: -26 + 0 * t
    cloud_10.scale = [30, 18, 1]
    cloud_10.can_leave = True
    cloud_10.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_10]

    cloud_5_node = sg.SceneGraphNode("cloud_5")
    cloud_5_node.childs = [Shapes[0]]
    textured_node.childs += [cloud_5_node]
    cloud_5 = Element(cloud_5_node)
    cloud_5.time = 0
    cloud_5.x_function = lambda t: -69 + 5 * t
    cloud_5.y_function = lambda t: 15 + 0 * t
    cloud_5.z_function = lambda t: -25 + 0 * t
    cloud_5.scale = [35, 25, 1]
    cloud_5.can_leave = True
    cloud_5.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_5]

    cloud_11_node = sg.SceneGraphNode("cloud_11")
    cloud_11_node.childs = [Shapes[1]]
    textured_node.childs += [cloud_11_node]
    cloud_11 = Element(cloud_11_node)
    cloud_11.time = 10
    cloud_11.x_function = lambda t: -69 + 5 * t
    cloud_11.y_function = lambda t: 15 + 0 * t
    cloud_11.z_function = lambda t: -24 + 0 * t
    cloud_11.scale = [40, 20, 1]
    cloud_11.can_leave = True
    cloud_11.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_11]

    #LOWER PLANE
    plane_node = sg.SceneGraphNode("plane")
    plane_node.childs = [Shapes[11]]
    textured_node.childs += [plane_node]
    plane = Element(plane_node)
    plane.time = 0
    plane.x_function = lambda t: -69 + 6 * t
    plane.y_function = lambda t: -15 + 0 * t
    plane.z_function = lambda t: -20 + 0 * t
    plane.scale = [15, 10, 1]
    plane.can_leave = True
    plane.exit_pos = [70, 0, 0]
    plane.rotation = [0, 0, np.pi]
    Environment_elements += [plane]

    # UPPER CLOUDS
    cloud_12_node = sg.SceneGraphNode("cloud_12")
    cloud_12_node.childs = [Shapes[3]]
    textured_node.childs += [cloud_12_node]
    cloud_12 = Element(cloud_12_node)
    cloud_12.time = 0
    cloud_12.x_function = lambda t: -69 + 2.5 * t
    cloud_12.y_function = lambda t: 10 + 0 * t
    cloud_12.z_function = lambda t: 39 + 0 * t
    cloud_12.scale = [40, 20, 1]
    cloud_12.can_leave = True
    cloud_12.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_12]

    cloud_13_node = sg.SceneGraphNode("cloud_13")
    cloud_13_node.childs = [Shapes[4]]
    textured_node.childs += [cloud_13_node]
    cloud_13 = Element(cloud_13_node)
    cloud_13.time = 0
    cloud_13.x_function = lambda t: -69 + 3.0 * t
    cloud_13.y_function = lambda t: -30 + 0 * t
    cloud_13.z_function = lambda t: 38 + 0 * t
    cloud_13.scale = [30, 17, 1]
    cloud_13.can_leave = True
    cloud_13.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_13]

    cloud_14_node = sg.SceneGraphNode("cloud_14")
    cloud_14_node.childs = [Shapes[1]]
    textured_node.childs += [cloud_14_node]
    cloud_14 = Element(cloud_14_node)
    cloud_14.time = 0
    cloud_14.x_function = lambda t: -69 + 3.5 * t
    cloud_14.y_function = lambda t: 30 + 0 * t
    cloud_14.z_function = lambda t: 37 + 0 * t
    cloud_14.scale = [35, 15, 1]
    cloud_14.can_leave = True
    cloud_14.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_14]

    cloud_15_node = sg.SceneGraphNode("cloud_15")
    cloud_15_node.childs = [Shapes[0]]
    textured_node.childs += [cloud_15_node]
    cloud_15 = Element(cloud_15_node)
    cloud_15.time = 0
    cloud_15.x_function = lambda t: -69 + 4.0 * t
    cloud_15.y_function = lambda t: -15 + 0 * t
    cloud_15.z_function = lambda t:  36 + 0 * t
    cloud_15.scale = [32, 27, 1]
    cloud_15.can_leave = True
    cloud_15.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_15]

    cloud_16_node = sg.SceneGraphNode("cloud_16")
    cloud_16_node.childs = [Shapes[2]]
    textured_node.childs += [cloud_16_node]
    cloud_16 = Element(cloud_16_node)
    cloud_16.time = 0
    cloud_16.x_function = lambda t: -69 + 4.5 * t
    cloud_16.y_function = lambda t: 0 + 0 * t
    cloud_16.z_function = lambda t: 35 + 0 * t
    cloud_16.scale = [35, 18, 1]
    cloud_16.can_leave = True
    cloud_16.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_16]

    cloud_17_node = sg.SceneGraphNode("cloud_5")
    cloud_17_node.childs = [Shapes[5]]
    textured_node.childs += [cloud_17_node]
    cloud_17 = Element(cloud_17_node)
    cloud_17.time = 0
    cloud_17.x_function = lambda t: -69 + 5 * t
    cloud_17.y_function = lambda t: 15 + 0 * t
    cloud_17.z_function = lambda t: 34 + 0 * t
    cloud_17.scale = [30, 15, 1]
    cloud_17.can_leave = True
    cloud_17.exit_pos = [70, 0, 0]
    Environment_elements += [cloud_17]

    # LOWER PLANE
    great_fox_node = sg.SceneGraphNode("great_fox")
    great_fox_node.childs = [Shapes[13]]
    textured_node.childs += [great_fox_node]
    great_fox = Element(great_fox_node)
    great_fox.time = 0
    great_fox.x_function = lambda t: 60 -3 * t
    great_fox.y_function = lambda t: 30 + 0 * t
    great_fox.z_function = lambda t: -20 + 1 * t
    great_fox.scale = [26, 13 , 1]
    great_fox.rotation = [np.pi/2, 0, 0]
    great_fox.can_leave = True
    great_fox.exit_pos = [60, 0, 0]
    Environment_elements += [great_fox]

    # JET
    jet_node = sg.SceneGraphNode("jet")
    jet_node.childs = [Shapes[12]]
    textured_node.childs += [jet_node]
    jet = Element(jet_node)
    jet.time = 0
    jet.x_function = lambda t: -60 + 10 * t
    jet.y_function = lambda t: -20 + 0 * t
    jet.z_function = lambda t: 3 + 0 * t
    jet.scale = [6, 2, 1]
    jet.rotation = [np.pi / 2, 0, 0]
    jet.can_leave = True
    jet.exit_pos = [60, 0, 0]
    Environment_elements += [jet]

    # DRONES
    dron_0_node = sg.SceneGraphNode("dron_0")
    dron_0_node.childs = [Shapes[10]]
    textured_node.childs += [dron_0_node]
    dron_0 = Element(dron_0_node)
    dron_0.time = 0
    dron_0.x_function = lambda t: 7.3912 + 2.2 + 0 * t
    dron_0.y_function = lambda t: -5.663 + 0 * t
    dron_0.z_function = lambda t: -6.323 + 1 + 0.7*np.sin(1.0 * t)
    dron_0.scale = [-1, 1, 1]
    dron_0.rotation = [np.pi / 2, 0, 0]
    dron_0.can_leave = False
    Environment_elements += [dron_0]

#Funcion para actualizar los objetos del ambiente
def UpdateEnvironment(delta):
    #se actualiza cada elemento con su metodo
    for element in Environment_elements:
        element.update(delta)
