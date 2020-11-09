import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

import track as tk
import easy_shaders as es
import transformations as tr
import basic_shapes as bs
import player as pl
import scene_graph as sg

#Variables globales
gpu_map = None # shape del minimapa
map_transform = None # transformacion del minimapa
gpu_player = None # shape del circulo que representa al auto del usuario
gpu_bot = None # shape del circulo que representa al auto - bot
backGround = None # shape del rectangulo de fondo para las vueltas
gpu_lap = None # shape del numero de vueltas
gpu_flag = None # shape del icono de la bandera
gpu_place = None # shape de la posicion

# Funcion que crea un circulo sin textura ni iluminacion
def createCircle(c1, c2, sides):
    vertices=[]
    indices=[]
    angle = 2*np.pi/sides
    vertices += [0, 0, 0, c1[0], c1[1], c1[2]]
    for v in range(sides):
        vertices += [0.5*np.cos(angle*v), 0.5*np.sin(angle*v), 0, c2[0], c2[1], c2[2]]

        if v < sides - 1:
            indices += [0, v+1, v+2]
        elif v == sides - 1:
            indices += [0, v+1, 1]
    return bs.Shape(vertices, indices)

# Se inicializan los elementos del hud
def SetupHUD():
    global gpu_map, map_transform, gpu_player, gpu_bot, backGround, gpu_lap, gpu_flag, gpu_place

    color1 = [26/255, 140/255, 255/255]
    color2 = [105/255, 216/255, 255/255]
    # Se crea el minimapa
    gpu_map = es.toGPUShape(tk.createTrack(200, 3, color1, color2))
    map_transform = tr.matmul([tr.translate(0.8,-0.68,-0.5), tr.scale(0.0112,0.025,0.02), tr.rotationX(-np.deg2rad(20)), tr.rotationZ(np.pi)])
    # Se crean los simbolos del auto del usuario y el bot
    gpu_player = es.toGPUShape(createCircle([255/255, 255/255, 255/255], [215/255, 215/255, 236/255], 10))
    gpu_bot = es.toGPUShape(createCircle([255/255, 21/255, 21/255], [170/255, 0/255, 0/255], 10))
    # Se crea el fondo del contador de vueltas
    backGround = es.toGPUShape(bs.createColorQuad(0.2, 0.2, 0.2))
    # Se crea el icono de la bandera
    gpu_flag = es.toGPUShape(bs.createTextureQuad("Sprites/hud/flag.png"), GL_REPEAT, GL_LINEAR)
    #Se crean las texturas de los numeros cel contador de vueltas
    gpu_lap = [
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_1.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_2.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_3.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_4.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_5.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_6.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_7.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_8.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/lap_9.png"), GL_REPEAT, GL_LINEAR)
    ]
    #Se crean las texturas que indican la posicion del usuario
    gpu_place = [
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/first.png"), GL_REPEAT, GL_LINEAR),
        es.toGPUShape(bs.createTextureQuad("Sprites/hud/second.png"), GL_REPEAT, GL_LINEAR)
    ]

# Se actualizan los elementos del hud
def UpdateHUD(pipeline, textured_pipeline):
    #Se dibujan los elementos del hud con transparencia sin texturas ni iluminacion
    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "transform"), 1, GL_TRUE, map_transform)
    pipeline.drawShape(gpu_map)
    backGround_transform = tr.matmul([tr.translate(-0.64, -0.87, -0.9), tr.scale(0.23, 0.17, 1)])
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "transform"), 1, GL_TRUE, backGround_transform)
    pipeline.drawShape(backGround)

    pos = pl.Player_Car.position
    player_transform = tr.matmul([tr.translate(0, 0, -0.1), map_transform, tr.translate(pos[0], pos[1], pos[2] - 0), tr.uniformScale(2.0)])
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "transform"), 1, GL_TRUE, player_transform )
    pipeline.drawShape(gpu_player)

    pos_bot = pl.Bot_Car.position
    bot_transform = tr.matmul(
        [tr.translate(0, 0, -0.1), map_transform, tr.translate(pos_bot[0], pos_bot[1], pos_bot[2] ), tr.uniformScale(2.0)])
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "transform"), 1, GL_TRUE, bot_transform)
    pipeline.drawShape(gpu_bot)

    # Se dibujan los elementos sin texturas ni iluminacion
    glUseProgram(textured_pipeline.shaderProgram)
    flag_transform = tr.matmul([tr.translate(-0.69, -0.87, -1), tr.scale(0.09, 0.15, 1)])
    glUniformMatrix4fv(glGetUniformLocation(textured_pipeline.shaderProgram, "transform"), 1, GL_TRUE, flag_transform)
    textured_pipeline.drawShape(gpu_flag)

    lap_transform = tr.matmul([tr.translate(-0.60, -0.87, -1), tr.scale(0.09, 0.15, 1)])
    glUniformMatrix4fv(glGetUniformLocation(textured_pipeline.shaderProgram, "transform"), 1, GL_TRUE, lap_transform)
    textured_pipeline.drawShape(gpu_lap[pl.Player_Car.lap - 1])

    #Se dibuja la posicion del usuario segun corresponda (primerop o segundo)
    if ((pl.Player_Car.lap > pl.Bot_Car.lap) or
            (pl.Player_Car.lap == pl.Bot_Car.lap and pl.Player_Car.t >= pl.Bot_Car.t)):
        #Si el usuario lleva mas vueltas dadas y tienen un t mayor
        place = 0
    else:
        place = 1

    place_transform = tr.matmul([tr.translate(-0.88, -0.8, -1), tr.scale(0.22, 0.34, 1)])
    glUniformMatrix4fv(glGetUniformLocation(textured_pipeline.shaderProgram, "transform"), 1, GL_TRUE, place_transform)
    textured_pipeline.drawShape(gpu_place[place])


