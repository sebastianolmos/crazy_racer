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
import track as tk

#Constantes para el auti manejado por el usuario
PLAYER_MAX_SPEED = 2 #Velocidad maxima del auto
PLAYER_MAX_REVERSE_SPEED = 1 #Velocidad maxima de retroseso
PLAYER_ACC = 0.005 #Aceleracion para avanzar
PLAYER_FRICTION = 0.015 #Friccion con la que se detiene
PLAYER_ROT_ACC = 1 #Aceleracion para rotar
PLAYER_ROT_MAX_SPEED = 0.8 #Maxima velocidad de rotacion
PLAYER_ROT_FRICTION = 2 #friccion para dejar de rotar

CAMERA_HEIGTH = 0.4 #altura de la camara con respecto a la pista
CAMERA_BACK_DISTANCE = 0.55 #distancia entre la camara y el auto proyectada en la pista

#Velocidad del bot
BOT_CAR_SPEED = 0.007

#variables que contendran los objetos auto
Player_Car = None
Bot_Car = None

#Clase Camara esferica para visualizar la escena completa
class SphericCamera:

    def __init__(self):
        self.center = np.array([0, 0, 0])

        self.phi_angle = 0.0
        self.theta_angle = - np.pi / 2
        self.eyeX = 0.0
        self.eyeY = 0.0
        self.eyeZ = 0.0
        self.viewPos = np.zeros(3)
        self.view = 0.0
        self.radius = 15
        self.up = np.array([0, 0, 1])

    def change_theta_angle(self, dt):
        self.theta_angle = (self.theta_angle + dt) % (np.pi * 2)

    def change_phi_angle(self, dt):
        self.phi_angle = (self.phi_angle + dt) % (np.pi * 2)

    def change_zoom(self, dr):
        if self.radius + dr > 0.1:
            self.radius += dr

    def update_view(self):
        self.eyeX = self.radius * np.sin(self.theta_angle) * np.cos(self.phi_angle) + self.center[0]
        self.eyeY = self.radius * np.sin(self.theta_angle) * np.sin(self.phi_angle) + self.center[1]
        self.eyeZ = self.radius * np.cos(self.theta_angle) + self.center[2]

        up_x = np.cos(self.theta_angle) * np.cos(self.phi_angle) * np.array([1, 0, 0])
        up_y = np.cos(self.theta_angle) * np.sin(self.phi_angle) * np.array([0, 1, 0])
        up_z = - np.sin(self.theta_angle) * np.array([0, 0, 1])
        self.up = up_x + up_y + up_z

        self.viewPos = np.array([self.eyeX, self.eyeY, self.eyeZ])

        self.view = tr.lookAt(
            self.viewPos,
            self.center,
            self.up
        )
        return self.view

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True

        self.is_up_pressed = False
        self.is_down_pressed = False
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_space_press = False
        self.is_z_pressed = False
        self.is_x_pressed = False
        self.is_w_pressed = False
        self.is_s_pressed = False
        self.is_a_pressed = False
        self.is_d_pressed = False
        self.global_camera = False
        self.back_camera = True
        # 0 = global_view
        # 1 = back_view
        # 2 = front_view
        # 3 = left_view
        # 4 = right_view
        self.camera_view = 1

        self.spheric_camera = SphericCamera()

    @property
    def camera(self):
        """ Get a camera reference from the controller object. """
        return self.spheric_camera

#Clase auto
class Car:
    def __init__(self):
        #Coordenadas de la pista parametrizada
        self.t = 0.994
        self.s = -0.6
        self.rotation = 0.143475 #rotacion del auto
        self.velocity = 0.0 #velocidad
        self.rot_speed = 0 #velocidad de rotacion
        self.transform_pos = None #referencia al nodo a realizar la traslacion
        self.transform_rot = None #referencia al nodo a realizar la rotacion
        self.track_width = 0.0

        self.position = 0 #posicion en el espacio
        #Vectores ortonormales
        self.tangente = 0
        self.binormal = 0
        self.normal = 0
        #vector que apunta en la direccion de movimiento
        self.forward = 0
        #rotacion de la rueda
        self.wheel_rotation = 0

        #arreglo para menejar en que vuelta se encuentra
        self.track_quarters = [True, False, False, False]
        #numero de vueltas
        self.lap = 1

    #metodo para controlar las vueltas
    def update_lap(self, last_t):
        #En cada vuelta se guarda si ha pasado por cada cuarto de pista
        if self.t < 0.25 and self.t > 0.00:
            self.track_quarters[0] = True
        elif self.t < 0.50 and self.t > 0.25:
            self.track_quarters[1] = True
        elif self.t < 0.75 and self.t > 0.50:
            self.track_quarters[2] = True
        elif self.t < 1.00 and self.t > 0.75 and self.track_quarters[2]==True:
            self.track_quarters[3] = True

        #Si cruza la linea de partida y ha pasdo por todos los cuartos, ha dado una vuelta
        if self.t < 0.1 and last_t > 0.9 and \
                self.track_quarters[0] == True and self.track_quarters[1] == True and \
                self.track_quarters[2] == True and self.track_quarters[3] == True:
            self.lap += 1
            self.t = 0
            self.track_quarters[0] = False
            self.track_quarters[1] = False
            self.track_quarters[2] = False
            self.track_quarters[3] = False
            #maximo de 9 vueltas
            if self.lap > 9:
                self.lap = 9

#Clase Face para guardar la informacion de los vertices, texturas para ambos autos y la normal a una cara
class Face:
    def __init__(self, vertexData, tx, n, n_text, text2):
        self.vertices = vertexData
        self.tx_coords = tx
        self.normal = n
        self.texture = n_text
        self.texture2 = text2

#Funcion Para crear la data del auto
def createDataCar():
    #vertices del auto, comparten caras
    v = [
        [-2.196251, -0.823580, 0.425225],  # v_0
        [-2.352234, -1.000000, -0.396911],  # v_1
        [-2.196251, 0.823579, 0.425225],  # v_2
        [-2.352234, 1.000000, -0.396911],  # v_3
        [1.818700, -1.000000, 0.192187],  # v_4
        [1.818696, -0.986844, -0.396911],  # v_5
        [1.818700, 1.000000, 0.192187],  # v_6
        [1.818000, 0.986844, -0.396911],  # v_7
        [2.267185, -0.527716, 0.042769],  # v_8
        [2.196633, -0.527716, -0.396911],  # v_9
        [0.299471, 0.675221, 0.900655],  # v_10
        [2.267185, 0.447083, 0.042769],  # v_11
        [0.299472, -0.675221, 0.900655],  # v_12
        [2.196633, 0.447083, -0.396911],  # v_13
        [-1.016603, -0.695153, 0.639356],  # v_14
        [-1.016603, 0.695153, 0.639356],  # v_15
        [-2.352234, -1.000000, 0.396911],  # v_16
        [-2.352234, 1.000000, 0.396911],  # v_17
        [-0.860620, -0.837610, 0.667670],  # v_18
        [-0.860621, 0.837610, 0.667670],  # v_19
        [-2.087752, -0.823580, -0.172497],  # v_20
        [-2.087752, 0.823579, -0.172497],  # v_21
        [-0.908104, -0.695153, 0.041634],  # v_22
        [-0.908104, 0.695153, 0.041634]  # v_23
    ]
    #Se cargan y se guardan las texturas para el auto manejado por el usuario
    textures = [
        saveTexture("Sprites/car/face_0.png", GL_REPEAT, GL_LINEAR),      # 0 - face 0
        saveTexture("Sprites/car/face_1_3.png", GL_REPEAT, GL_LINEAR),    # 1 - face 1, 3
        saveTexture("Sprites/car/face_2.png", GL_REPEAT, GL_LINEAR),      # 2 - face 2
        saveTexture("Sprites/car/face_4_7.png", GL_REPEAT, GL_LINEAR),    # 3 - face 4, 7
        saveTexture("Sprites/car/face_5_8.png", GL_REPEAT, GL_LINEAR),    # 4 - face 5, 8
        saveTexture("Sprites/car/face_6.png", GL_REPEAT, GL_LINEAR),      # 5 - face 6
        saveTexture("Sprites/car/face_9.png", GL_REPEAT, GL_LINEAR),      # 6 - face 9
        saveTexture("Sprites/car/face_10-13.png", GL_REPEAT, GL_LINEAR),  # 7 - face 10-13
        saveTexture("Sprites/car/face_14.png", GL_REPEAT, GL_LINEAR),     # 8 - face 14
        saveTexture("Sprites/car/face_15_17.png", GL_REPEAT, GL_LINEAR),  # 9 - face 15, 17
        saveTexture("Sprites/car/face_16.png", GL_REPEAT, GL_LINEAR),     # 10 - face 16
        saveTexture("Sprites/car/face_18.png", GL_REPEAT, GL_LINEAR),     # 11 - face 18
    ]
    #Se cargan y se guardan las texturas del auto manejado por el bot
    textures2 = [
        saveTexture("Sprites/bot/face_0.png", GL_REPEAT, GL_LINEAR),  # 0 - face 0
        saveTexture("Sprites/bot/face_1_3.png", GL_REPEAT, GL_LINEAR),  # 1 - face 1, 3
        saveTexture("Sprites/bot/face_2.png", GL_REPEAT, GL_LINEAR),  # 2 - face 2
        saveTexture("Sprites/bot/face_4_7.png", GL_REPEAT, GL_LINEAR),  # 3 - face 4, 7
        saveTexture("Sprites/bot/face_5_8.png", GL_REPEAT, GL_LINEAR),  # 4 - face 5, 8
        saveTexture("Sprites/bot/face_6.png", GL_REPEAT, GL_LINEAR),  # 5 - face 6
        saveTexture("Sprites/bot/face_9.png", GL_REPEAT, GL_LINEAR),  # 6 - face 9
        saveTexture("Sprites/bot/face_10-13.png", GL_REPEAT, GL_LINEAR),  # 7 - face 10-13
        saveTexture("Sprites/bot/face_14.png", GL_REPEAT, GL_LINEAR),  # 8 - face 14
        saveTexture("Sprites/bot/face_15_17.png", GL_REPEAT, GL_LINEAR),  # 9 - face 15, 17
        saveTexture("Sprites/bot/face_16.png", GL_REPEAT, GL_LINEAR),  # 10 - face 16
        saveTexture("Sprites/bot/face_18.png", GL_REPEAT, GL_LINEAR),  # 11 - face 18
    ]

    #Se crean las cras, con la iformacion de sus vertices, texturas y normales
    faces = [
        # Face(vertices,
        #      coordenadas de las texturas por cada vertice,
        #      vector normal, textura guardada del auto manejado por el usuario, textura guardada del bot)
        Face([v[4], v[8], v[11], v[6], v[10], v[12]],
             [[0.0958, 0.8773], [0.2328, 1], [0.7672, 1], [0.9042, 0.8773], [0.8133, 0], [0.1867, 0]],
             [0.4017, 0.0000, 0.9158], textures[0], textures2[0]),  # Face 0
        Face([v[5], v[9], v[8], v[4]],
             [[0, 1], [1, 1], [1, 0.2], [0, 0.1]],
             [0.7419, -0.6678, -0.0594], textures[1], textures2[1]),  # Face 1
        Face([v[9], v[13], v[11], v[8]],
             [[0, 1], [1, 1], [1, 0.23], [0, 0.23]],
             [0.9874, 0.0000, -0.1584], textures[2], textures2[2]),  # Face 2
        Face([v[13], v[7], v[6], v[11]],
             [[1, 1], [0, 1], [0, 0.1], [1, 0.2]],
             [0.7916, 0.6078, -0.0626], textures[1], textures2[1]),  # Face 3
        Face([v[7], v[3], v[17], v[6]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [0.0013, 1.0000, -0.0095], textures[3], textures2[3]),  # Face 4
        Face([v[6], v[17], v[10]],
             [[0, 1], [1, 1], [0.413, 0.11]],
             [0.0249, 0.8948, 0.4458 ], textures[4], textures2[4]),  # Face 5
        Face([v[3], v[1], v[16], v[17]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-1.0000, 0.0000, 0.0000], textures[5], textures2[5]),  # Face 6
        Face([v[1], v[5], v[4], v[16]],
             [[1, 1], [0, 1], [0, 0], [1, 0]],
             [0.0013, -1.0000, -0.0095], textures[3], textures2[3]),  # Face 7
        Face([v[16], v[4], v[12]],
             [[1, 1], [0, 1], [0.413, 0.11]],
             [0.0249, -0.8948, 0.4458 ], textures[4], textures2[4]),  # Face 8
        Face([v[19], v[18], v[12], v[10]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.1969, -0.0000, 0.9804], textures[6], textures2[6]),  # Face 9
        Face([v[15], v[14], v[18], v[19]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.1786, 0.0000, 0.9839], textures[7], textures2[7]),  # Face 10
        Face([v[19], v[17], v[2], v[15]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.1786, 0.0000, 0.9839], textures[7], textures2[7]),  # Face 11
        Face([v[17], v[16], v[0], v[2]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.1786, 0.0000, 0.9839], textures[7], textures2[7]),  # Face 12
        Face([v[16], v[18], v[14], v[0]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.1786, 0.0000, 0.9839], textures[7], textures2[7]),  # Face 13
        Face([v[21], v[20], v[22], v[23]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.1786, 0.0000, 0.9839], textures[8], textures2[8]),  # Face 14
        Face([v[21], v[23], v[15], v[2]],
             [[0, 1], [1, 1], [1, 0], [0, 0.3724]],
             [-0.1048, -0.9943, -0.0190], textures[9], textures2[9]),  # Face 15
        Face([v[23], v[22], v[14], v[15]],
             [[0, 1], [1, 1], [1, 0], [0, 0]],
             [-0.9839, -0.0000, -0.1786], textures[10], textures2[10]),  # Face 16
        Face([v[22], v[20], v[0], v[14]],
             [[0, 1], [1, 1], [1, 00.3724], [0, 0]],
             [-0.1048, 0.9943, -0.0190], textures[9], textures2[9]),  # Face 17
        Face([v[20], v[21], v[2], v[0]],
             [[0, 1], [1, 1], [0, 1], [0, 0]],
             [0.9839, 0.0000, 0.1786], textures[11], textures2[11]),  # Face 18
        Face([v[1], v[3], v[7], v[13], v[9], v[5]],
             [[0.15, 1], [0.85, 1], [1, 0.85], [0.85, 0], [0.15, 0], [0, 0.85]],
             [0.0000, 0.0000, -1.0000 ], textures[7], textures2[7])  # Face 19
    ]
    #Se retorna una arreglo de caras
    return faces

#Funcion que crea una rueda (con texturas e iluminacion)
def createNormalWheel(sides, width):
    #Funcion similar al ade cracion de un  cilindro con normales
    w_vertices = []
    w_indices = []
    t_vertices = []
    t_indices = []

    w_vertices += [0, width/2, 0, 0.5, 0.5, 0, 1, 0]
    w_vertices += [0, -width/2, 0, 0.5, 0.5, 0, -1, 0]
    rad = 0.5
    counter = 0
    increase = 0.5
    angle = 2*np.pi/sides
    for i in range(sides):
        w_vertices += [-np.cos(i*angle)*0.5, width / 2, np.sin(i*angle)*0.5, 0.5+rad*np.cos(i*angle), 0.5+rad*np.sin(i*angle),
                       0, 1, 0]
        w_vertices += [-np.cos(i * angle) * 0.5, -width / 2, np.sin(i * angle) * 0.5, 0.5 + rad * np.cos(i * angle), 0.5 + rad * np.sin(i * angle),
                       0, -1, 0]

        t_vertices += [-np.cos(i * angle) * 0.5, width / 2, np.sin(i * angle) * 0.5, counter, 0,
                       -np.cos(i*angle), 0, np.sin(i*angle)]
        t_vertices += [-np.cos(i * angle) * 0.5, -width / 2, np.sin(i * angle) * 0.5, counter, 1,
                       -np.cos(i*angle), 0, np.sin(i*angle)]

        counter += increase
        t_indices += [2*i+0, 2*i+1, 2*i+3, 2*i+3, 2*i+2, 2*i+0]

        w_indices += [0, 2 * i, 2 * i + 2]
        w_indices += [1, 2 * i + 1, 2 * i + 3]

    w_indices += [0, len(w_vertices)/8-2, 2]
    w_indices += [1, len(w_vertices)/8-1, 3]
    t_vertices += [-np.cos(0) * 0.5, width / 2, np.sin(0) * 0.5, counter, 0,
                   -np.cos(0), 0, np.sin(0)]
    t_vertices += [-np.cos(0) * 0.5, -width / 2, np.sin(0) * 0.5, counter, 1,
                   -np.cos(0), 0, np.sin(0)]

    #parametros del material
    ka = [0.3, 0.3, 0.3]
    kd = [0.6, 0.6, 0.6]
    ks = [0.8, 0.8, 0.8]
    #Se guardan en nodos distintos los lados y las tapas, para ocupar diferentes texturas
    case_shape = LightShape(w_vertices, w_indices, ka, kd, ks, "Sprites/car/wheel.png")
    tire_shape = LightShape(t_vertices, t_indices, ka, kd, ks, "Sprites/car/tire.png")
    wheel_node = sg.SceneGraphNode("wheel")
    case_node = sg.SceneGraphNode("case")
    case_node.childs += [toGPULightShape(case_shape, GL_REPEAT, GL_LINEAR)]
    wheel_node.childs += [case_node]
    tire_node = sg.SceneGraphNode("tire")
    tire_node.childs += [toGPULightShape(tire_shape, GL_REPEAT, GL_LINEAR)]
    wheel_node.childs += [tire_node]

    return wheel_node

#Funcion que crea una rueda (con texturas y sin iluminacion)
def createWheel(sides, width):
    # Funcion similar al ade cracion de un  cilindro
    w_vertices = []
    w_indices = []
    t_vertices = []
    t_indices = []

    w_vertices += [0, width/2, 0, 0.5, 0.5]
    w_vertices += [0, -width/2, 0, 0.5, 0.5]
    rad = 0.5
    counter = 0
    increase = 0.5
    angle = 2*np.pi/sides
    for i in range(sides):
        w_vertices += [-np.cos(i*angle)*0.5, width / 2, np.sin(i*angle)*0.5, 0.5+rad*np.cos(i*angle), 0.5+rad*np.sin(i*angle)]
        w_vertices += [-np.cos(i * angle) * 0.5, -width / 2, np.sin(i * angle) * 0.5, 0.5 + rad * np.cos(i * angle), 0.5 + rad * np.sin(i * angle)]

        t_vertices += [-np.cos(i * angle) * 0.5, width / 2, np.sin(i * angle) * 0.5, counter, 0]
        t_vertices += [-np.cos(i * angle) * 0.5, -width / 2, np.sin(i * angle) * 0.5, counter, 1]

        counter += increase
        t_indices += [2*i+0, 2*i+1, 2*i+3, 2*i+3, 2*i+2, 2*i+0]

        w_indices += [0, 2 * i, 2 * i + 2]
        w_indices += [1, 2 * i + 1, 2 * i + 3]

    w_indices += [0, len(w_vertices)/5-2, 2]
    w_indices += [1, len(w_vertices)/5-1, 3]
    t_vertices += [-np.cos(0) * 0.5, width / 2, np.sin(0) * 0.5, counter, 0]
    t_vertices += [-np.cos(0) * 0.5, -width / 2, np.sin(0) * 0.5, counter, 1]

    # Se guardan en nodos distintos los lados y las tapas, para ocupar diferentes texturas
    case_shape = bs.Shape(w_vertices, w_indices, "Sprites/car/wheel.png")
    tire_shape = bs.Shape(t_vertices, t_indices, "Sprites/car/tire.png")
    wheel_node = sg.SceneGraphNode("wheel")
    case_node = sg.SceneGraphNode("case")
    case_node.childs += [es.toGPUShape(case_shape, GL_REPEAT, GL_LINEAR)]
    wheel_node.childs += [case_node]
    tire_node = sg.SceneGraphNode("tire")
    tire_node.childs += [es.toGPUShape(tire_shape, GL_REPEAT, GL_LINEAR)]
    wheel_node.childs += [tire_node]

    return wheel_node

#funcion que crea un nodo con el auto principal con todas sus partes
def createCarShape():

    f = createDataCar() # data de las caras
    car_node = sg.SceneGraphNode("car") #nodo principal del auto

    wheel_set = sg.SceneGraphNode("wheels_set") # nodo conjunto de ruedas

    #Se crea el nodo con la rueda con iluminacion
    wheel_node = sg.SceneGraphNode("wheel")
    wheel_node.transform = tr.uniformScale(0.9)
    wheel_node.childs += [createNormalWheel(10, 0.5)]
    scaled_wheel = sg.SceneGraphNode("scaled_wheel")
    scaled_wheel.childs += [wheel_node]

    #Se crean los nodos para las cuatro ruedas del auto

    rotated_wheel_0 = sg.SceneGraphNode("rotated_wheel_0")
    rotated_wheel_0.transform = tr.translate(1.27, 1.0, -0.4)
    rotated_wheel_0.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_0]

    rotated_wheel_1 = sg.SceneGraphNode("rotated_wheel_2")
    rotated_wheel_1.transform = tr.translate(1.27, -1.0, -0.4)
    rotated_wheel_1.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_1]

    rotated_wheel_2 = sg.SceneGraphNode("rotated_wheel_3")
    rotated_wheel_2.transform = tr.translate(-1.67, 1.0, -0.4)
    rotated_wheel_2.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_2]

    rotated_wheel_3 = sg.SceneGraphNode("rotated_wheel_4")
    rotated_wheel_3.transform = tr.translate(-1.67, -1.0, -0.4)
    rotated_wheel_3.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_3]

    car_node.childs += [wheel_set]

    #se crea un nodo por cada cara, para ocupar diferentes texturas y se anaden al nodo principal
    for face in f:
        temp_vertices = []
        temp_indices = []
        for v in range(len(face.vertices)):
            #por cada cara se agregan vertices
            temp_v = face.vertices[v]
            temp_tx = face.tx_coords[v]
            temp_vertices += [temp_v[0], temp_v[1], temp_v[2], temp_tx[0], temp_tx[1], face.normal[0], face.normal[1], face.normal[2]]
            if v < len(face.vertices) - 2:
                temp_indices += [0, v + 1, v + 2]
        #Valores del material
        temp_ka = [0.3, 0.3, 0.33]
        temp_kd = [0.4, 0.4, 0.4]
        temp_ks = [1.0, 1.0, 1.0]
        temp_shape = LightShape(temp_vertices, temp_indices, temp_ka, temp_kd, temp_ks)
        temp_node = sg.SceneGraphNode("car_face")
        temp_node.childs += [toGPUTexturedLightShape(temp_shape, face.texture)]
        car_node.childs += [temp_node]

    return car_node

#funcion que crea un nodo con el auto-bot con todas sus partes
def createCarBot():
    f = createDataCar()  # data de las caras
    car_node = sg.SceneGraphNode("car") #nodo principal del auto

    wheel_set = sg.SceneGraphNode("wheels_set") # nodo conjunto de ruedas

    #Se crea el nodo con la rueda
    wheel_node = sg.SceneGraphNode("wheel")
    wheel_node.transform = tr.uniformScale(0.9)
    wheel_node.childs += [createWheel(10, 0.5)]
    scaled_wheel = sg.SceneGraphNode("scaled_wheel")
    scaled_wheel.childs += [wheel_node]

    #Se crean los nodos para las cuatro ruedas del auto

    rotated_wheel_0 = sg.SceneGraphNode("rotated_wheel_0")
    rotated_wheel_0.transform = tr.translate(1.27, 1.0, -0.4)
    rotated_wheel_0.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_0]

    rotated_wheel_1 = sg.SceneGraphNode("rotated_wheel_2")
    rotated_wheel_1.transform = tr.translate(1.27, -1.0, -0.4)
    rotated_wheel_1.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_1]

    rotated_wheel_2 = sg.SceneGraphNode("rotated_wheel_3")
    rotated_wheel_2.transform = tr.translate(-1.67, 1.0, -0.4)
    rotated_wheel_2.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_2]

    rotated_wheel_3 = sg.SceneGraphNode("rotated_wheel_4")
    rotated_wheel_3.transform = tr.translate(-1.67, -1.0, -0.4)
    rotated_wheel_3.childs += [scaled_wheel]
    wheel_set.childs += [rotated_wheel_3]

    car_node.childs += [wheel_set]

    # se crea un nodo por cada cara, para ocupar diferentes texturas y se anaden al nodo principal
    for face in f:
        temp_vertices = []
        temp_indices = []
        for v in range(len(face.vertices)):
            # por cada cara se agregan vertices
            temp_v = face.vertices[v]
            temp_tx = face.tx_coords[v]
            temp_vertices += [temp_v[0], temp_v[1], temp_v[2], temp_tx[0], temp_tx[1]]
            if v < len(face.vertices) - 2:
                temp_indices += [0, v + 1, v + 2]
        temp_shape = bs.Shape(temp_vertices, temp_indices)
        temp_node = sg.SceneGraphNode("car_face")
        temp_node.childs += [toGPUTexturedShape(temp_shape, face.texture2)]
        car_node.childs += [temp_node]

    return car_node

# funcion para entregar el input a la camara de visualizacion global
def InputToCamera(control, camera, delta):
    if control.is_left_pressed:
        camera.change_phi_angle(-2 * delta)

    if control.is_right_pressed:
        camera.change_phi_angle( 2 * delta)

    if control.is_up_pressed:
        camera.change_theta_angle( 2 * delta)

    if control.is_down_pressed:
        camera.change_theta_angle(-2 * delta)

    if control.is_x_pressed:
        camera.change_zoom(5 * delta)

    if control.is_z_pressed:
        camera.change_zoom(-5 * delta)

#Funcion para actualizar la camara segun su tipo de visualizacion actual
def CameraUpdate(controller, delta):
    camera = controller.camera
    #Tipo de camara controlado por un valor entero
    # 0 = global_view
    # 1 = back_view
    # 2 = front_view
    # 3 = left_view
    # 4 = right_view
    InputToCamera(controller, camera, delta)
    if controller.camera_view == 0: # vista global
        view = controller.camera.update_view()
        viewPos = camera.viewPos
    #Vista en tercera persona
    else:
        #Se pondera el vector binormal por el vector unitario en z, para que lacamra rote un poco
        up = Player_Car.binormal*0.6 + vector3_Z*0.4
        #Vista por detras
        if controller.camera_view == 1:
            camera_axis = -Player_Car.forward
            viewPos = Player_Car.position + CAMERA_BACK_DISTANCE * camera_axis + CAMERA_HEIGTH * Player_Car.binormal
            center = Player_Car.position + Player_Car.binormal * 0.25
        # Vista por delante
        elif controller.camera_view == 2:
            camera_axis = Player_Car.forward
            viewPos = Player_Car.position + 0.45 * camera_axis + 0.15 * Player_Car.binormal
            center = Player_Car.position + Player_Car.binormal * 0.0
        # Vista lateral por la izquierda
        elif controller.camera_view == 3:
            rot_normal = np.cross(Player_Car.forward, Player_Car.binormal)
            viewPos = Player_Car.position - 0.3 * rot_normal + 0.15 * Player_Car.binormal
            center = Player_Car.position + Player_Car.binormal * 0.05
        # Vista lateral por la derecha
        else:
            rot_normal = np.cross(Player_Car.forward, Player_Car.binormal)
            viewPos = Player_Car.position + 0.3 * rot_normal + 0.15 * Player_Car.binormal
            center = Player_Car.position + Player_Car.binormal * 0.05

        #matriz de vista
        view = tr.lookAt(
            viewPos,
            center,
            up
        )
    #Finalmente se entrega la posicion de la camara y la matriz de vista
    return viewPos, view

#Funcion para inicializar los autos
def SetupCars(textured_light_node, botScene_node):
    global Player_Car, Bot_Car

    #Auto manejado por el usuario
    Player_Car = Car()
    Player_Car.track_width = tk.Track_Curve.width # valor del ancho de la pista
    #Se crea la figura del auto, se añade a un nodo y se escala
    scene_node = sg.SceneGraphNode("player_car")
    scene_node.transform = tr.uniformScale(0.1)
    scene_node.childs += [createCarShape()]
    #Se agrega a un nodo, en el que se aplicara las rotaciones
    scaled_node = sg.SceneGraphNode("scaled_car")
    scaled_node.childs += [scene_node]
    #Se agrega a un nodo, en el que se aplicara la traslacion a la posicion
    rotated_node = sg.SceneGraphNode("rotated_car")
    rotated_node.childs += [scaled_node]

    # Se asignan los valores inciales a los parametros del auto
    p, t, b, n = tk.GetTrackData(Player_Car.t, Player_Car.s)
    Player_Car.position = p
    Player_Car.tangente = t
    Player_Car.binormal = b
    Player_Car.normal = n
    Player_Car.forward = t
    Player_Car.s = -0.6
    Player_Car.rotation = 0.143475
    #Se anade al grafo de escena
    textured_light_node.childs += [rotated_node]

    # Auto - BOT
    Bot_Car = Car()
    Bot_Car.track_width = tk.Track_Curve.width  # valor del ancho de la pista
    #Se crea la figura del auto, se añade a un nodo y se escala
    bot_node = sg.SceneGraphNode("bot_car")
    bot_node.transform = tr.uniformScale(0.1)
    bot_node.childs += [createCarBot()]
    #Se agrega a un nodo, en el que se aplicara las rotaciones
    scaled_bot = sg.SceneGraphNode("scaled_bot")
    scaled_bot.childs += [bot_node]
    #Se agrega a un nodo, en el que se aplicara la traslacion a la posicion
    rotated_bot = sg.SceneGraphNode("rotated_bot")
    rotated_bot.childs += [scaled_bot]

    # Se asignan los valores inciales a los parametros del auto
    Bot_Car.position = p
    Bot_Car.tangente = t
    Bot_Car.binormal = b
    Bot_Car.normal = n
    Bot_Car.forward = t
    Bot_Car.s = 0.6
    Bot_Car.rotation = 0.143475
    #Se anade al grafo de escena
    botScene_node.childs += [rotated_bot]

#Funcion para actualizar el auto manejador por el usuario
def UpdatePlayer(controller, textured_light_node, delta):
    #Control de la velocidad con aceleraciones, hasta alcanzar una maxima velocidad
    if controller.is_w_pressed:
        if Player_Car.velocity > PLAYER_MAX_SPEED:
            Player_Car.velocity = PLAYER_MAX_SPEED
        else:
            Player_Car.velocity += PLAYER_ACC
    elif controller.is_s_pressed:
        if Player_Car.velocity < -PLAYER_MAX_REVERSE_SPEED:
            Player_Car.velocity = -PLAYER_MAX_REVERSE_SPEED
        else:
            Player_Car.velocity -= PLAYER_ACC * 2.5
    #Si no hay input, el auto se frena
    else: #DESACELERAR
        if Player_Car.velocity <= PLAYER_FRICTION and Player_Car.velocity >= -PLAYER_FRICTION:
            Player_Car.velocity = 0
        elif Player_Car.velocity > 0:
            Player_Car.velocity -= PLAYER_FRICTION
        elif Player_Car.velocity < 0:
            Player_Car.velocity += PLAYER_FRICTION


    #Se hace la rotacion dependiendo del signo de la velocidad del auto con aceleraciones
    if controller.is_a_pressed and Player_Car.velocity > 0 or \
            controller.is_d_pressed and Player_Car.velocity < 0:
        if Player_Car.rot_speed > PLAYER_ROT_MAX_SPEED:
            Player_Car.rot_speed = PLAYER_ROT_MAX_SPEED
        else:
            Player_Car.rot_speed += PLAYER_ROT_ACC * delta
    elif controller.is_a_pressed and Player_Car.velocity < 0 or \
            controller.is_d_pressed and Player_Car.velocity > 0:
        if Player_Car.rot_speed < -PLAYER_ROT_MAX_SPEED:
            Player_Car.rot_speed = -PLAYER_ROT_MAX_SPEED
        else:
            Player_Car.rot_speed -= PLAYER_ROT_ACC * delta
    # Desaceleracion si no hay input
    else:
        if Player_Car.rot_speed <= PLAYER_ROT_FRICTION * delta*2 and Player_Car.rot_speed >= -PLAYER_ROT_FRICTION * delta*2:
            Player_Car.rot_speed = 0
        elif Player_Car.rot_speed > 0:
            Player_Car.rot_speed -= PLAYER_ROT_FRICTION * delta *2
        elif Player_Car.rot_speed < 0:
            Player_Car.rot_speed += PLAYER_ROT_FRICTION * delta *2

    #Se aplica la velocidad de rotacion al angulo de rotacion
    Player_Car.rotation = (Player_Car.rotation + Player_Car.rot_speed * delta) % (2*np.pi)

    #Se calcula el angulo con respecto al eje normal
    tan_in_xy = normalize(np.array([Player_Car.tangente[0], Player_Car.tangente[1], 0]))
    if Player_Car.tangente[2] > 0:
        rot_y = np.arccos(np.dot(tan_in_xy, Player_Car.tangente))
    elif Player_Car.tangente[2] == 0:
        rot_y = 0
    else:
        rot_y = -np.arccos(np.dot(Player_Car.tangente, tan_in_xy))
    #Se calcula el angulo con respecto al eje tangente
    normal_in_xy = normalize(np.array([Player_Car.normal[0], Player_Car.normal[1], 0]))
    if Player_Car.normal[2] > 0:
        rot_x = np.arccos(np.dot(normal_in_xy, Player_Car.normal))
    elif Player_Car.normal[2] == 0:
        rot_x = 0
    else:
        rot_x = -np.arccos(np.dot(Player_Car.normal, normal_in_xy))
    #Se aplican todas las rotaciones, primero la rotacion en la que se mueve el auto, luego los angulos para la inclinacion
    car_rotation = tr.matmul([tr.rotationA(rot_y, Player_Car.normal), tr.rotationA(-rot_x, Player_Car.tangente), tr.rotationZ(Player_Car.rotation)])

    #Vector con la direccion de movimiento, transformando el vector unitario x con la rotacion del auto
    Player_Car.forward = transformVector(vector3_X, car_rotation)
    #candidato a posicion moviendo la posicion actual segun el vector forward
    candidate_pos = Player_Car.position + Player_Car.forward*Player_Car.velocity*delta
    #se calculan los candidatos a coordenadas de la pista con la posicion candidato
    t_candidate, s_candidate = tk.nearest_track_coord(Player_Car, candidate_pos)
    #se guarda el ultimo t
    last_t = Player_Car.t
    #se actualizan las coordenadas de la pista si los candidatos a coordenadas no se sale de la pista
    if np.abs(s_candidate) < Player_Car.track_width*0.45:
        Player_Car.t = t_candidate
        Player_Car.s = s_candidate
        #Se actualizan la posicion y los vectores ortogonales
        p, t, b, n = tk.GetTrackData(Player_Car.t, Player_Car.s)
        Player_Car.position = p
        Player_Car.tangente = t
        Player_Car.binormal = b
        Player_Car.normal = n
    else:
        #se supone que ha chocado en el borde, por lo que el auto se detiene
        Player_Car.velocity=0
        p, t, b, n = tk.GetTrackData(Player_Car.t, Player_Car.s)
        Player_Car.position = p
        Player_Car.tangente = t
        Player_Car.binormal = b
        Player_Car.normal = n

    #referencias a los nodos en los que se aplican las transformaciones
    rotated_car = textured_light_node.childs[1]
    scaled_car = rotated_car.childs[0]

    #Rotacion a las ruedas del auto segun la velocidad de este
    wheels = scaled_car.childs[0].childs[0].childs[0].childs
    for wheel in wheels:
        scaled_wheel = wheel.childs[0].childs[0]
        scaled_wheel.transform = tr.rotationY(Player_Car.wheel_rotation)
    wheel_speed = Player_Car.velocity * 0.1
    Player_Car.wheel_rotation = (Player_Car.wheel_rotation + wheel_speed)

    #Traslacion para posicionar el auto
    rotated_car.transform = tr.translate(Player_Car.position[0], Player_Car.position[1], Player_Car.position[2])
    #Rotaciones del auto y un ajuste en la altura
    scaled_car.transform = tr.matmul([car_rotation, tr.translate(0,0, 0.07)])

    #Controlador de la cantidad de vueltas en la pista del auto
    Player_Car.update_lap(last_t)

#Funcion para actualizar el
def UpdateBot(node, delta):
    #Se guarda el ultimo t
    last_t = Bot_Car.t
    #se mueve el auto a velocidad constante, manteniendo el s
    Bot_Car.t += BOT_CAR_SPEED*delta
    Bot_Car.t %= 1

    #Se actualiza la data con las coordenadas de la pista
    p, t, b, n = tk.GetTrackData(Bot_Car.t, Bot_Car.s)
    Bot_Car.position = p
    Bot_Car.tangente = t
    Bot_Car.normal = n
    #referencias a los nodos a transformar
    rotated_bot = node.childs[0]
    scaled_bot = rotated_bot.childs[0]

    #Se calculan los angulos que definen la rotacion e inclinaciones del auto

    tan_in_xy = normalize(np.array([Bot_Car.tangente[0], Bot_Car.tangente[1], 0]))
    if Bot_Car.tangente[2] > 0:
        rot_y = np.arccos(np.dot(tan_in_xy, Bot_Car.tangente))
    elif Bot_Car.tangente[2] == 0:
        rot_y = 0
    else:
        rot_y = -np.arccos(np.dot(Bot_Car.tangente, tan_in_xy))

    normal_in_xy = normalize(np.array([Bot_Car.normal[0], Bot_Car.normal[1], 0]))
    if Bot_Car.normal[2] > 0:
        rot_x = np.arccos(np.dot(normal_in_xy, Bot_Car.normal))
    elif Bot_Car.normal[2] == 0:
        rot_x = 0
    else:
        rot_x = -np.arccos(np.dot(Bot_Car.normal, normal_in_xy))

    if tan_in_xy[1] > 0:
        rot_z = np.arccos(np.dot(vector3_X, tan_in_xy))
    elif Bot_Car.tangente[1] == 0:
        rot_z = 0
    else:
        rot_z = -np.arccos(np.dot(tan_in_xy, vector3_X))

    #Matriz con las rotaciones
    car_rotation = tr.matmul([tr.rotationA(rot_y, Bot_Car.normal), tr.rotationA(-rot_x, Bot_Car.tangente), tr.rotationZ(rot_z)])

    # Traslacion para posicionar el auto
    rotated_bot.transform = tr.translate(p[0], p[1], p[2])
    # Rotaciones del auto y un ajuste en la altura
    scaled_bot.transform = tr.matmul([car_rotation, tr.translate(0, 0, 0.07)])
    # Controlador de la cantidad de vueltas en la pista del bot
    Bot_Car.update_lap(last_t)


    # Controlador de la cantidad de vueltas en la pista del auto
    Player_Car.update_lap(last_t)

