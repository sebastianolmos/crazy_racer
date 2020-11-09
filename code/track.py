import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import math

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as ls
import scene_graph as sg

from operations import *

#Velocidad de rotacion de las turbinas
FAN_ROT_SPEED = 0.6

#Variables globales
Track_Curve = None #Curva que define la pista
fan_rotation = 0

#Objeto nodo con informacion para crear la RNU Spline
class Node:
    def __init__(self, position):
        self.position = np.array(position)
        self.velocity = np.zeros(3)
        self.distance = 0
        self.binormal = None

#Objeto para asignar la informacion a los nodos
class Curve_Nodes:
    def __init__(self, nodes):
        self.nodes = nodes
        self.total_distance = 0
        self.width = 0

    #Metodo para calcular el largo de la pista
    def calcule_distance(self):
        for index in range(len(self.nodes)):
            nodeA = self.nodes[index]
            if index == (len(self.nodes) - 1):
                nodeB = self.nodes[0]
            else:
                nodeB = self.nodes[index + 1]
            pos_a = nodeA.position
            pos_b = nodeB.position
            self.nodes[index].distance = np.sqrt(
                (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2 + (pos_a[2] - pos_b[2]) ** 2)
            self.total_distance += self.nodes[index].distance

    #Metodo que establece las relaciones de vecindad y la velocidad para cada nodo
    def Rounded_Setup(self):
        for index in range(len(self.nodes)):
            #nodos vecinos
            if index == 0:
                prevNode = self.nodes[len(self.nodes) - 1]
                nextNode = self.nodes[index + 1]
            elif index == (len(self.nodes) - 1):
                prevNode = self.nodes[index - 1]
                nextNode = self.nodes[0]
            else:
                prevNode = self.nodes[index - 1]
                nextNode = self.nodes[index + 1]
            #vectores apuntando a los vecinos
            p2_1 = (nextNode.position - self.nodes[index].position) / self.nodes[index].distance
            p0_1 = (prevNode.position - self.nodes[index].position) / prevNode.distance
            #Se calcula la velocidad
            bisectriz = normalize(p2_1 + p0_1)
            normal = normalize(np.cross(p2_1, p0_1))
            vel = np.cross(bisectriz, normal)
            self.nodes[index].velocity = vel
            #Se calcula la binormal a cada nodo
            final_normal = np.cross(p2_1, p0_1)
            if normal[2]<0:
                final_normal = np.cross(p0_1, p2_1)
            self.nodes[index].binormal = normalize(final_normal)

#Se crean los nodos que daran forma a la pista
def CreateTrackNodes(transform):
    #SE DEFINEN LOS NODOS
    nodes = []
    nodes += [[ -1, 2, -2]] #0
    nodes += [[ 2.35, 1.75, -2]] #1
    nodes += [[ 3.5, -2.0, 0]] #2
    nodes += [[ 5.5, -0.5,-1.0]] #3
    nodes += [[ 3.5, 0.5,-2]] #4
    nodes += [[ 2.5, -1.0, -2]] #5
    nodes += [[ 2.5, -3.5, -2]] #6
    nodes += [[ 1.75, -5.25, 2]] #7
    nodes += [[ 1, -3.5, -2]] #8
    nodes += [[ 0.8, -1.4, -2]] #9
    nodes += [[-1.8, -0.75, 0]] #10
    nodes += [[-1.8, 1.15, -0.67]] #11
    nodes += [[-0.25, 0.7, -1.33]] #12
    nodes += [[-1, -0.75, -2]] #13
    nodes += [[-2.85, -0.65,-2]] #14
    nodes += [[-3.8, 4.5, 2]] #15
    nodes += [[-4.65, 1.15, -2]]  #16
    transformated_nodes = []
    for node in nodes:
        t_node = tr.matmul([transform, np.array([node[0], node[1], node[2], 1]).T])
        d3_node = np.array([t_node[0], t_node[1], t_node[2]])/ t_node[3]
        transformated_nodes += [Node(d3_node)]
    return transformated_nodes

#Funcion para obtener la posicion de la curva dado la posicion de dos nodos y sus tangentes/velocidades
def GetPositionOnCubic(node0_pos, vel0, node1_pos, vel1, time):
    t = np.array([time**3, time**2, time, 1])
    H = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
    G = np.array([node0_pos, node1_pos, vel0, vel1])
    position = np.matmul(np.matmul(t, H), G)
    return position

#Funcion para obtener la posicion de un punto de la curva dado un tiempo entre 0 y 1
def GetPosition(time):
    nodes = Track_Curve.nodes
    maxDistance = Track_Curve.total_distance

    time = time%1
    distance = time * maxDistance
    currentDistance = 0.0
    i = 0
    while (currentDistance + nodes[i].distance < distance and i < len(nodes) - 1):
        currentDistance += nodes[i].distance
        i += 1

    i_next = i +1
    if i == len(nodes) - 1:
        i_next = 0

    t = distance - currentDistance
    t = t/nodes[i].distance
    startVel = nodes[i].velocity * nodes[i].distance
    endVel = nodes[i_next].velocity * nodes[i].distance
    return GetPositionOnCubic(nodes[i].position, startVel, nodes[i_next].position, endVel, t)

#Funcion para obtener la data(posicion y vectores ortogonales) de la curva entre dos nodos
def GetDataOnCubic(node0, vel0, node1, vel1, time):
    node0_pos = node0.position
    node1_pos = node1.position
    t = np.array([time**3, time**2, time, 1])
    t_prima = np.array([3*(time**2), 2*time, 1, 0])
    H = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
    G = np.array([node0_pos, node1_pos, vel0, vel1])

    position = np.matmul(np.matmul(t, H), G)

    tangente = normalize(np.matmul(np.matmul(t_prima, H), G))

    binormal = Lerp(node0.binormal, node1.binormal, time)
    normal = normalize(np.cross(tangente, binormal))
    binormal =normalize(np.cross(normal, tangente)) #Vector binormal corregido
    return position, tangente, binormal, normal

#Funcion para obtener la data(posicion y vectores ortogonales) de un punto de la curva dado un tiempo entre 0 y 1
def GetData(time):
    nodes = Track_Curve.nodes
    maxDistance = Track_Curve.total_distance

    time = time%1
    distance = time * maxDistance
    currentDistance = 0.0
    i = 0
    while (currentDistance + nodes[i].distance < distance and i < len(nodes) - 1):
        currentDistance += nodes[i].distance
        i += 1

    i_next = i +1
    if i == len(nodes) - 1:
        i_next = 0

    t = distance - currentDistance
    t = t/nodes[i].distance
    startVel = nodes[i].velocity * nodes[i].distance
    endVel = nodes[i_next].velocity * nodes[i].distance
    return GetDataOnCubic(nodes[i], startVel, nodes[i_next], endVel, t)

#Funcion para crear una pista sin iluminacion ni texturas, utilizado para el minimapa de la pista
def createTrack(N, width, c1, c2):
    points = np.linspace(0.0, 1.0, N)
    vertices = []
    indices = []
    #Se crea la pista generando dos vertices a los lados de la curva, definidos por el vector normal a cada punto
    for p in range(len(points)):
        pos, tan, bi, norm = GetData(points[p])
        if points[p] < 0.005: #pintar de otro color el punto de partida
            vertices += [pos[0]-norm[0]*width/2, pos[1]-norm[1]*width/2, pos[2]-norm[2]*width/2, 1, 0, 0]
            vertices += [pos[0]+norm[0]*width/2, pos[1]+norm[1]*width/2, pos[2]+norm[2]*width/2, 1, 0, 0]
        else:
            vertices += [pos[0] - norm[0] * width / 2, pos[1] - norm[1] * width / 2, pos[2] - norm[2] * width / 2,
                         c1[0], c1[1], c1[2]]
            vertices += [pos[0] + norm[0] * width / 2, pos[1] + norm[1] * width / 2, pos[2] + norm[2] * width / 2,
                         c2[0], c2[1], c2[2]]
        if p < len(points)-1:
            indices += [2*p + 2, 2*p + 0, 2*p + 1]
            indices += [2*p + 2, 2*p + 3, 2*p + 1]

    return bs.Shape(vertices, indices)

#Funcion para crear la pista con exturas e iluminacion
def createCompleteTrack(N, width, fileName):
    points = np.linspace(0.0, 1.0, N)
    vertices = [] #vertices de la pista
    start_v = [] #vertices de la linea de partida
    indices = [] #indices de la partida
    start_i = [] #indices de la linea de partida
    complete_node = sg.SceneGraphNode("complete_track")
    #contador de texturar para cuadrar las texturas en la pista
    texture_counter = 0
    start_tx_counter = 0
    last_p = 0
    draw_start = True
    for p in range(len(points)):
        pos, tan, bi, norm = GetData(points[p])
        if points[p] < 0.004: #Se crea la linea de partida
            start_v += [pos[0] - norm[0] * width / 2, pos[1] - norm[1] * width / 2, pos[2] - norm[2] * width / 2] #vertice izquierdo
            start_v += [0, start_tx_counter, bi[0], bi[1], bi[2]] # posicion de las texturas y normales
            start_v += [pos[0] + norm[0] * width / 2, pos[1] + norm[1] * width / 2, pos[2] + norm[2] * width / 2] #vertice derecho
            start_v += [1, start_tx_counter, bi[0], bi[1], bi[2]] # posicion de las texturas y normales
            start_tx_counter += 0.4
            start_i += [2 * p + 2, 2 * p + 0, 2 * p + 1]
            start_i += [2 * p + 2, 2 * p + 3, 2 * p + 1]
            last_p = p
            draw_start = True
        else: # se crea el resto de la pista
            vertices += [pos[0]-norm[0]*width/2, pos[1]-norm[1]*width/2, pos[2]-norm[2]*width/2] #vertice izquierdo
            vertices += [0, texture_counter, bi[0], bi[1], bi[2]] # posicion de las texturas y normales
            vertices += [pos[0]+norm[0]*width/2, pos[1]+norm[1]*width/2, pos[2]+norm[2]*width/2] #vertice derecho
            vertices += [1, texture_counter, bi[0], bi[1], bi[2]] # posicion de las texturas y normales
            texture_counter += 0.2
            if p < len(points)-1:
                p = p - (last_p+1)
                indices += [2*p + 2, 2*p + 0, 2*p + 1]
                indices += [2*p + 2, 2*p + 3, 2*p + 1]
            if draw_start: # se rellena parte de la pista que habia quedado vacia
                start_v += [pos[0] - norm[0] * width / 2, pos[1] - norm[1] * width / 2, pos[2] - norm[2] * width / 2]
                start_v += [0, start_tx_counter, bi[0], bi[1], bi[2]]
                start_v += [pos[0] + norm[0] * width / 2, pos[1] + norm[1] * width / 2, pos[2] + norm[2] * width / 2]
                start_v += [1, start_tx_counter, bi[0], bi[1], bi[2]]
                draw_start = False

    # Se añade un banner al comienzo de la pista
    pos, tan, bi, norm = GetData(points[0])
    b_width = 3.8
    banner_v = [
        pos[0] - norm[0] * b_width/2, pos[1] - norm[1] * b_width/2, pos[2] - norm[2] * b_width/2, 0, 1,
        pos[0] + norm[0] * b_width/2, pos[1] + norm[1] * b_width/2, pos[2] + norm[2] * b_width/2, 1, 1,
        pos[0] + norm[0] * b_width/2 + bi[0]*2, pos[1] + norm[1] * b_width/2 + bi[1]*2, pos[2] + norm[2] * b_width/2 + bi[2]*2, 1, 0,
        pos[0] - norm[0] * b_width/2 + bi[0]*2, pos[1] - norm[1] * b_width/2 + bi[1]*2, pos[2] - norm[2] * b_width/2 + bi[2]*2, 0, 0
    ]
    banner_i = [0, 1, 2, 2, 3, 0]


    Ka = [0.3, 0.32, 0.32]
    Kd = [0.6, 0.6, 0.6]
    Ks = [0.8, 0.8, 0.8]

    #se añaden las distintas shapes a nodos
    track_shape = LightShape(vertices, indices  , Ka, Kd, Ks, fileName)
    track_node = sg.SceneGraphNode("track")
    track_node.childs += [toGPULightShape(track_shape, GL_REPEAT, GL_LINEAR)]
    complete_node.childs += [track_node]
    start_shape = LightShape(start_v, start_i, Ka, Kd, Ks, "Sprites/start_line.png")
    start_node = sg.SceneGraphNode("start")
    start_node.childs += [toGPULightShape(start_shape, GL_REPEAT, GL_LINEAR)]
    complete_node.childs += [start_node]
    banner_shape = bs.Shape(banner_v, banner_i, "Sprites/banner.png")
    banner_node = sg.SceneGraphNode("banner")
    banner_node.childs += [es.toGPUShape(banner_shape, GL_REPEAT, GL_LINEAR)]
    #Finalmente se entrega un nodo con la pista completa (con iluminacion y texturas) y otro con el banner (solo texturas)
    return complete_node, banner_node

#funcion para crear la parte inferior y lateral de la pista, para darle grosor
def createTrackBack(N, width, height):
    points = np.linspace(0.0, 1.0, N)
    vertices = []
    indices = []
    #se añaden los vertices, (shape sin texturas ni iluminacion)
    for p in range(len(points)):
        pos, tan, binormal, normal = GetData(points[p])
        '''
            Surface
        3 ------------- 2    ^ binormal
        |               |    |
        |               |    |
        |     Back      |    o -----> normal
        0 ------------- 1    tangente
        '''
        vertex_3 = [pos[0] - normal[0]  *width / 2, pos[1] - normal[1] * width / 2, pos[2] - normal[2] * width / 2]
        vertex_2 = [pos[0] + normal[0] * width / 2, pos[1] + normal[1] * width / 2, pos[2] + normal[2] * width / 2]
        vertex_0 = [vertex_3[0] - binormal[0] * height, vertex_3[1] - binormal[1] * height, vertex_3[2] - binormal[2] * height]
        vertex_1 = [vertex_2[0] - binormal[0] * height, vertex_2[1] - binormal[1] * height,vertex_2[2] - binormal[2] * height]

        vertices += [vertex_0[0], vertex_0[1], vertex_0[2], 0.15, 0.15, 0.15]
        vertices += [vertex_1[0], vertex_1[1], vertex_1[2], 0.05, 0.05, 0.05]
        vertices += [vertex_2[0], vertex_2[1], vertex_2[2], 0.15, 0.15, 0.15]
        vertices += [vertex_3[0], vertex_3[1], vertex_3[2], 0.1, 0.1, 0.1]

        if p < len(points)-1:
            #Left side
            indices += [4 * p + 0, 4 * p + 3, 4 * p + 7]
            indices += [4 * p + 7, 4 * p + 4, 4 * p + 0]
            #Lower side
            indices += [4 * p + 0, 4 * p + 4, 4 * p + 5]
            indices += [4 * p + 5, 4 * p + 1, 4 * p + 0]
            #Right side
            indices += [4 * p + 1, 4 * p + 5, 4 * p + 6]
            indices += [4 * p + 6, 4 * p + 2, 4 * p + 1]

    return bs.Shape(vertices, indices)

#Funcion para obtener la data en un punto de la pista dado un t y s
def GetTrackData(time, side):
    time = time % 1
    pos, tangente, binormal, normal = GetData(time)

    space_pos = np.array([pos[0]+normal[0]*side, pos[1]+normal[1]*side, pos[2]+normal[2]*side])
    return space_pos, tangente, binormal, normal

#funcion para obtener el posicion en la curva (t) mas cercano a un punto del espacio
def nearest_t(car, next_pos):
    movement = next_pos - car.position
    #se proyecta el vector de movimiento al vector tangente
    delta_time = np.dot(movement, car.tangente) / Track_Curve.total_distance
    final_t = (car.t+delta_time)
    #se corrige elt para los casos bordes
    if final_t < 0:
        final_t = (1 + final_t) % 1
    else:
        final_t = final_t % 1
    return final_t

#funcion para obtener el desplazamiento a la curva (s) dado un t
def nearest_s(car, next_pos, t):
    curve_pos, tangente, binormal, normal = GetData(t)

    pos = curve_pos + car.s*normal
    A = binormal[0]
    B = binormal[1]
    C = binormal[2]
    D = (A*pos[0] + B*pos[1] + C*pos[2])
    lamb = np.abs((D - A*next_pos[0] - B*next_pos[1] - C*next_pos[2]) / (A**2 + B**2 + C**2))
    x = next_pos[0] + A * lamb
    y = next_pos[1] + B * lamb
    z = next_pos[2] + C * lamb
    plane_pos = np.array([x, y, z])
    curvePos_planePos =plane_pos - car.position
    side = np.dot(normal, curvePos_planePos)  + car.s
    return side

#funcion para obtener las coordenadas de la pista mas cercano a un punto del espacio
def nearest_track_coord(car, next_pos):
    near_t = nearest_t(car, next_pos)
    near_s = nearest_s(car, next_pos, near_t)
    return near_t, near_s

#Funcionm para crear una turbina
def createFanShape(blades):
    v_fan_0 = [ 0.3, 0.1, 0]
    v_fan_1 = [ 0.3,-0.1, 0]
    v_fan_2 = [ 2.5,-0.4, 0]
    v_fan_3 = [ 2.5, 0.4, 0]
    color1 = np.array([0.0, 0.0, 0.0])
    color2 = np.array([0.4, 0.4, 0.4])
    shape_0 = createRectangle(v_fan_0, v_fan_1, v_fan_2, v_fan_3, color1, color2)
    blade_angle = np.deg2rad(30)
    rot_blade_vertices = rotateShape("X", blade_angle, shape_0.vertices)

    fan_vertices = []
    fan_indices = []
    angle_bt_blades = 2*np.pi/blades
    vertex_counter = 0
    for b in range(blades):
        fan_vertices += rotateShape("Z", b*angle_bt_blades, rot_blade_vertices)
        for i in range(len(shape_0.indices)):
            fan_indices += [vertex_counter + shape_0.indices[i]]
        vertex_counter += len(rot_blade_vertices)//6

    sphere_shape = createColorSphere(13, 13, color2, color1, vertex_counter)
    fan_vertices += sphere_shape.vertices
    fan_indices += sphere_shape.indices

    return bs.Shape(fan_vertices, fan_indices)

#Funcion para crear el soporte de una turbina, dados los vectores ortoganeles de un punto de la curva que crea la pista
def createFanAttachment(tangente, binormal, position, width, joint, fan, sides = 6, radio = 0.1):
    normal = np.cross(tangente, binormal)
    color1 = np.array([0.0, 0.0, 0.0])
    color2 = np.array([0.4, 0.4, 0.4])

    vertices = []
    indices = []

    pos_left = position - normal*(width/2 -radio)
    pos_right = position + normal*(width/2 -radio)
    if pos_left[2] < pos_right[2]:
        lower_pos = np.array([position[0], position[1], pos_left[2]])
    else:
        lower_pos = np.array([position[0], position[1], pos_right[2]])
    pos_middle = lower_pos - vector3_Z*(joint)
    pos_top = lower_pos - vector3_Z*(joint)
    pos_bottom = pos_top - vector3_Z*fan
    tan_in_xy = normalize(np.array([tangente[0], tangente[1], 0]))
    normal_in_xy = normalize(np.array([normal[0], normal[1], 0]))
    angle = 2*np.pi/sides
    for i in range(sides):
        left = pos_left - tangente * np.cos(angle * i)*radio + normal * np.sin(angle * i)*radio
        left_bot = pos_middle - tan_in_xy * np.cos(angle * i)*radio + normal_in_xy * np.sin(angle * i)*radio
        right = pos_right - tangente * np.cos(angle * i)*radio - normal * np.sin(angle * i)*radio
        right_bot = pos_middle - tan_in_xy * np.cos(angle * i) * radio - normal_in_xy * np.sin(angle * i) * radio
        top = pos_top - tan_in_xy * np.cos(angle * i)*radio + normal_in_xy * np.sin(angle * i)*radio
        bottom = pos_bottom - tan_in_xy * np.cos(angle * i)*radio + normal_in_xy * np.sin(angle * i)*radio
        s_coef =  np.abs(sides//2 - i)/(sides//2)
        color = color2 * (1 - s_coef) + color1 * (s_coef)

        vertices += [left[0], left[1], left[2], color[0], color[0], color[0],
                     left_bot[0], left_bot[1], left_bot[2], color[0], color[0], color[0],
                     right[0], right[1], right[2], color[0], color[0], color[0],
                     right_bot[0], right_bot[1], right_bot[2], color[0], color[0], color[0],
                     top[0], top[1], top[2], color[0], color[0], color[0],
                     bottom[0], bottom[1], bottom[2], color[0], color[0], color[0]]
        if i < sides -1:
            indices += [6*i + 7, 6*i + 1, 6*i + 0, 6*i + 0, 6*i + 6, 6*i +7,
                        6*i + 3, 6*i + 9, 6*i + 8, 6*i + 8, 6*i + 2, 6*i +3,
                        6*i + 11, 6*i + 5, 6*i + 4, 6*i + 4, 6*i + 10, 6*i +11]
        else:
            indices += [1, 6*i + 1, 6*i + 0, 6*i + 0, 0, 1,
                        6*i + 3, 3, 2, 2, 6*i + 2, 6*i +3,
                        5, 6*i + 5, 6*i + 4, 6*i + 4, 4, 5]
    return bs.Shape(vertices, indices)

#Funcion para inicializar la pista
def SetupTrack(normal_node, textured_node, textured_light_node):
    global Track_Curve, Binormal_Points

    #transformacion con la que se crea la pista
    track_transform = tr.matmul([tr.translate(0, 0, 0), tr.scale(3, 3, 3)])
    track_nodes = CreateTrackNodes(track_transform) # se crean los nodos de la curva

    Track_Curve = Curve_Nodes(track_nodes) # se crea el conjunto de nodos
    Track_Curve.calcule_distance() # se calcula el largo de la pista
    Track_Curve.Rounded_Setup() # se establecen los parametros para crear la RNU Spline
    Track_Curve.width = 3 #  ancho de la pista
    # Se ajustan manualmente algunos vectores binormales de los nodos
    Track_Curve.nodes[6].binormal = (vector3_Z + vector3_Y) /2
    Track_Curve.nodes[8].binormal = (vector3_Z + vector3_Y) /2
    Track_Curve.nodes[9].binormal = vector3_Z
    Track_Curve.nodes[16].binormal = (vector3_X - vector3_Y + vector3_Z)/3
    Track_Curve.nodes[7].binormal = normalize(Track_Curve.nodes[7].binormal*0.70 + vector3_Y*0.30)

    #Se crean los elementos de la pista
    track_node, banner_node = createCompleteTrack(500, 3, "Sprites/track.jpg")
    trackBack_shape = createTrackBack(250, 3, 0.5)
    gpuBack = es.toGPUShape(trackBack_shape)

    # Se agregan a sus respectivos grafos de escena
    textured_light_node.childs += [track_node]

    back_node = sg.SceneGraphNode("back_track")
    back_node.childs += [gpuBack]
    normal_node.childs += [back_node]

    textured_node.childs += [banner_node]

    # Se establecen los puntos en la pista donde se ubicaran las turbinas
    #                |time | joint|fan|radio|
    fan_positions = [[0,      1.5, 0.7, 0.2],
                     [0.1895, 1.5, 0.7, 0.2],
                     [0.245,  1.5, 0.7, 0.2],
                     [0.3156, 1.5, 0.7, 0.2],
                     [0.392,  4.0, 3.0, 0.3],
                     [0.5116, 1.5, 0.7, 0.2],
                     [0.694,  1.5, 0.7, 0.2],
                     [0.825,  2.0, 1.2, 0.2],
                     [0.8984, 1.7, 1.7, 0.2],
                     [0.9468, 0.7, 0.7, 0.2]]
    gpu_fan = es.toGPUShape(createFanShape(6)) #shape de la turbina
    sides = 10
    fans_set = sg.SceneGraphNode("fans_set")
    #Se crean las turbinas sujetadas a la pista
    for f in fan_positions:
        t = f[0]
        joint = f[1]
        fan_dist = f[2]
        rad = f[3]
        temp_pos, temp_t, temp_b, temp_n = GetData(t)
        temp_p = temp_pos - temp_b * 0.3
        temp_left = temp_p -temp_n*(1.5-rad)
        temp_right = temp_p + temp_n * (1.5 - rad)
        if temp_left[2] < temp_right[2]:
            lower_pos = np.array([temp_p[0], temp_p[1], temp_left[2]])
        else:
            lower_pos = np.array([temp_p[0], temp_p[1], temp_right[2]])
        temp_support = es.toGPUShape(createFanAttachment(temp_t, temp_b, temp_p, 3, joint, fan_dist, sides, rad))
        fan_pos = lower_pos - vector3_Z*(joint + fan_dist)
        fan_node = sg.SceneGraphNode("fan")
        fan_node.childs += [gpu_fan]
        rotatedFan_node = sg.SceneGraphNode("rotated_fan")
        rotatedFan_node.transform = tr.translate(fan_pos[0], fan_pos[1], fan_pos[2])
        rotatedFan_node.childs += [fan_node]
        support_node = sg.SceneGraphNode("support")
        support_node.childs += [temp_support]
        fixed_fan = sg.SceneGraphNode("fixed_fan")
        fixed_fan.childs += [support_node]
        fixed_fan.childs += [rotatedFan_node]
        fans_set.childs += [fixed_fan]
    normal_node.childs += [fans_set]

#Funcion para actualizar la pista
def UpdateTrack(normal_node, delta):
    global fan_rotation
    #Se rotan las turbinas
    fan_array = normal_node.childs[1].childs
    for fan in fan_array:
        fan_rotation = (fan_rotation + delta*FAN_ROT_SPEED)%(2*np.pi)
        to_rotate = fan.childs[1].childs[0]
        to_rotate.transform = tr.rotationZ(fan_rotation)
