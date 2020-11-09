# Sebastian Olmos

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import math

#Se importan archivos dados en el curso
import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as ls
import scene_graph as sg

#Se importan otros archivos creados
from operations import *
import player as pl
import track as tk
import environment as env
import display as hud

#Cantidad de frames
FPS = 60
#Dimensiones de la ventana, proporcion ideal de 16/9
WIDTH = 1440
HEIGHT = 810
FLYING_SPEED = 1  #Velocidad con que oscila la pista
FLYING_AMPLITUDE = 2

#variables globales para contener grafos de escenas
World = None # Escena de los objetos sin texturas ni iluminacionj
Textured_world = None # Escena de los objetos con texturas, sin iluminacion
Textured_light_world = None # Escena con los objetos que tienen texturas e iluminacion
bot_node = None # Escena con la figura del robot que solo tiene texturas
flying_pos = 0 # altura de la pista

#Se referencia al controlador del player
controller = pl.Controller()
#Se recibe el input
def on_key(window, key, scancode, action, mods):

    global controller

    if key == glfw.KEY_SPACE:
        if action == glfw.PRESS:
            controller.fillPolygon = not controller.fillPolygon

    if key == glfw.KEY_ESCAPE:
        if action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    elif key == glfw.KEY_LEFT_CONTROL:
        if action == glfw.PRESS:
            controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_V:
        if action == glfw.PRESS:
            #Se la vista de la camara
            controller.camera_view = (controller.camera_view + 1) % 5

    if key == glfw.KEY_UP:
        if action == glfw.PRESS:
            controller.is_up_pressed = True
        elif action == glfw.RELEASE:
            controller.is_up_pressed = False

    if key == glfw.KEY_DOWN:
        if action == glfw.PRESS:
            controller.is_down_pressed = True
        elif action == glfw.RELEASE:
            controller.is_down_pressed = False

    if key == glfw.KEY_RIGHT:
        if action == glfw.PRESS:
            controller.is_right_pressed = True
        elif action == glfw.RELEASE:
            controller.is_right_pressed = False

    if key == glfw.KEY_LEFT:
        if action == glfw.PRESS:
            controller.is_left_pressed = True
        elif action == glfw.RELEASE:
            controller.is_left_pressed = False

    if key == glfw.KEY_Z:
        if action == glfw.PRESS:
            controller.is_z_pressed = True
        elif action == glfw.RELEASE:
            controller.is_z_pressed = False

    if key == glfw.KEY_X:
        if action == glfw.PRESS:
            controller.is_x_pressed = True
        elif action == glfw.RELEASE:
            controller.is_x_pressed = False

    if key == glfw.KEY_W:
        if action == glfw.PRESS:
            controller.is_w_pressed = True
        elif action == glfw.RELEASE:
            controller.is_w_pressed = False

    if key == glfw.KEY_S:
        if action == glfw.PRESS:
            controller.is_s_pressed = True
        elif action == glfw.RELEASE:
            controller.is_s_pressed = False

    if key == glfw.KEY_A:
        if action == glfw.PRESS:
            controller.is_a_pressed = True
        elif action == glfw.RELEASE:
            controller.is_a_pressed = False

    if key == glfw.KEY_D:
        if action == glfw.PRESS:
            controller.is_d_pressed = True
        elif action == glfw.RELEASE:
            controller.is_d_pressed = False

#Funcion que inicializa el programa
def Setup():
    global World, Textured_world, Textured_light_world, bot_node

    # Se crean los grafos de escenadd
    World = sg.SceneGraphNode("normal")
    Textured_world = sg.SceneGraphNode("textured")
    bot_node = sg.SceneGraphNode("bot_node")
    Textured_light_world = sg.SceneGraphNode("textured_light")

    #Se inicializan los objetos
    env.SetupEnvironment(Textured_world) #decoraciones de la escena
    tk.SetupTrack(World, Textured_world, Textured_light_world) #pista
    pl.SetupCars(Textured_light_world, bot_node) #Auto del usuario y bot
    hud.SetupHUD() #elementos del hud

#Funcion que actualiza el programa cada frame, recibiendo los diferentes pipelines necesarios para dibujar la escena
def Update(normal_pipeline, textured_pipeline, light_pipeline, alpha_pipeline, texture_hud_pipeline, delta):
    global flying_pos

    # Setting up the projection transform
    projection = tr.perspective(80, float(WIDTH) / float(HEIGHT), 0.1, 100)
    # Filling or not the shapes depending on the controller state
    if (controller.fillPolygon):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    #Se actualizan los objetos y parametros del programa
    pl.UpdatePlayer(controller, Textured_light_world, delta)
    pl.UpdateBot(bot_node, delta)
    tk.UpdateTrack(World, delta)
    env.UpdateEnvironment(delta)

    #Se aplican transformaciones a la pista y los autos para que oscilen
    flying_pos = (flying_pos + delta*FLYING_SPEED) % (2*np.pi)
    levitate_tr = tr.translate(0, 0, FLYING_AMPLITUDE*np.sin(flying_pos))
    World.transform = levitate_tr
    Textured_world.childs[-1].transform = levitate_tr
    Textured_light_world.transform = levitate_tr
    bot_node.transform = levitate_tr
    pl.Player_Car.position = pl.Player_Car.position + np.array([0,0, FLYING_AMPLITUDE*np.sin(flying_pos)])

    #Se obtiene la vista vista del player
    viewPos, view = pl.CameraUpdate(controller, delta)

    #Se dibuja la escena normal, objetos sin texturas ni iluminacion
    glUseProgram(normal_pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(normal_pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    glUniformMatrix4fv(glGetUniformLocation(normal_pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    sg.drawSceneGraphNode(World, normal_pipeline, "model")

    #Se dibuja la escena con texturas e iluminacion
    glUseProgram(light_pipeline.shaderProgram)
    glUniform3f(glGetUniformLocation(light_pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(light_pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(light_pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    light_pos = [0, 4, 3]
    glUniform3f(glGetUniformLocation(light_pipeline.shaderProgram, "lightPosition"), light_pos[0], light_pos[1], light_pos[2])
    glUniform3f(glGetUniformLocation(light_pipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
    glUniform1ui(glGetUniformLocation(light_pipeline.shaderProgram, "shininess"), 200)

    glUniform1f(glGetUniformLocation(light_pipeline.shaderProgram, "constantAttenuation"), 0.8)
    glUniform1f(glGetUniformLocation(light_pipeline.shaderProgram, "linearAttenuation"), 0) #Se anulan la dependecia a la distancia de la iluminacion
    glUniform1f(glGetUniformLocation(light_pipeline.shaderProgram, "quadraticAttenuation"), 0) #Se anulan la dependecia a la distancia de la iluminacion
    glUniformMatrix4fv(glGetUniformLocation(light_pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    glUniformMatrix4fv(glGetUniformLocation(light_pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    drawLightedNode(Textured_light_world, light_pipeline, "model")

    # Se dibuja la escena con texturas sin iluminacion, y el bot
    glUseProgram(textured_pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(textured_pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    glUniformMatrix4fv(glGetUniformLocation(textured_pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    sg.drawSceneGraphNode(bot_node, textured_pipeline, "model")
    sg.drawSceneGraphNode(Textured_world, textured_pipeline, "model")

    #Se dibujan los elementos del hud
    hud.UpdateHUD(alpha_pipeline, texture_hud_pipeline)

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit() 

    window = glfw.create_window(WIDTH, HEIGHT, "Crazy Racer!", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    #Pieplines necesarios
    normalPipeline = es.SimpleModelViewProjectionShaderProgram()
    texturePipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleTexturePhongShaderProgram()
    texture2dPipeline = es.SimpleTextureTransformShaderProgram()
    alpha2dPipeline = AlphaTransformShaderProgram() #pipeline con transparencia
    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    #Inicializacion
    Setup()

    t0 = glfw.get_time()
    time_counter = 0

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        #Controlador de frames
        if time_counter > 1/FPS:
            # Clearing the screen in both, color and depth
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            #Actualizacion
            Update(normalPipeline, texturePipeline, lightPipeline, alpha2dPipeline, texture2dPipeline, time_counter)
            time_counter = 0
            # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
            glfw.swap_buffers(window)
        time_counter += dt


    glfw.terminate()