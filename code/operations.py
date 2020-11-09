import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

import numpy as np
import transformations as tr
import scene_graph as sg
import easy_shaders as es
import basic_shapes as bs

SIZE_IN_BYTES = 4

#Vectores unitarios utiles
vector2_X = np.array([1,0])
vector2_Y = np.array([0,1])
vector3_X = np.array([1, 0, 0])
vector3_Y = np.array([0, 1, 0])
vector3_Z = np.array([0, 0, 1])

#GPUShape que contiene informacion del material (ka, kd y ks)
class GPULightShape(es.GPUShape):
    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ebo = 0
        self.texture = 0
        self.size = 0
        self.Ka = [0.3, 0.3, 0.3]
        self.Kd = [0.3, 0.3, 0.3]
        self.Ks = [0.3, 0.3, 0.3]

#Shape que contiene informacion del material (ka, kd y ks)
class LightShape:
    def __init__(self, vertices, indices, ka, kd, ks, textureFileName=None):
        self.vertices = vertices
        self.indices = indices
        self.textureFileName = textureFileName
        self.Ka = ka
        self.Kd = kd
        self.Ks = ks

#Funcion para calcular la magnitud de un vector
def magnitude(vector):
    mag = None
    if len(vector) == 2:
        mag = np.sqrt(vector[0]**2 + vector[1]**2)
    elif len(vector) == 3:
        mag = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    return mag

#Funcion para normalizar un vector
def normalize(vector):
    normalized = None
    if len(vector) == 2:
        normalized = vector / np.sqrt(vector[0]**2 + vector[1]**2)
    elif len(vector) == 3:
        normalized = vector / np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    return normalized

#Funcion para transformar vector2 a vector3
def vector3(vector2):
    return np.array([vector2[0], vector2[1], 0])

#Funcion para transformar vector3 a vector2
def vector2(vector3):
    return np.array([vector3[0], vector3[1]])

#Funcion para aplicar una transformacion a un vector/vertice
def transformVector(vector3, transform):
    homogeneus = np.array([vector3[0], vector3[1], vector3[2], 1])
    transformed_v = tr.matmul([transform, homogeneus.T])
    final_vector = np.array([transformed_v[0], transformed_v[1], transformed_v[2]]) / transformed_v[3]
    return final_vector

#Funcion interpolacion lineal
def Lerp(A, B, C):
    return (A*(1-C) + B*C)

#Funcion para cargar y guardar una textura
def saveTexture(image_fileName, wrapMode, filterMode):
    texture = glGenTextures(1)
    es.textureSimpleSetup(texture, image_fileName, wrapMode, filterMode)
    return texture

#Funcion para convertir a una GPUShape con una textura cargada
def toGPUTexturedShape(shape, texture):
    vertices = shape.vertices
    index = shape.indices
    vertexData = np.array(vertices, dtype=np.float32)
    indices = np.array(index, dtype=np.uint32)
    # Here the new shape will be stored
    gpuShape = es.GPUShape()

    gpuShape.size = len(indices)
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    gpuShape.texture = texture
    return gpuShape

#Funcion para convertir a una GPUShape con iluminacion
def toGPULightShape(shape, wrapMode=None, filterMode=None):
    assert isinstance(shape, LightShape)

    vertexData = np.array(shape.vertices, dtype=np.float32)
    indices = np.array(shape.indices, dtype=np.uint32)

    # Here the new shape will be stored
    gpuShape = GPULightShape()

    gpuShape.size = len(shape.indices)
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)
    gpuShape.Ka = shape.Ka
    gpuShape.Ks = shape.Ks
    gpuShape.Kd = shape.Kd

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    if shape.textureFileName != None:
        assert wrapMode != None and filterMode != None

        gpuShape.texture = glGenTextures(1)
        es.textureSimpleSetup(gpuShape.texture, shape.textureFileName, wrapMode, filterMode)

    return gpuShape

#Funcion para convertir a una GPUShape con una textura cargada e iluminacion
def toGPUTexturedLightShape(shape, texture):
    assert isinstance(shape, LightShape)

    vertexData = np.array(shape.vertices, dtype=np.float32)
    indices = np.array(shape.indices, dtype=np.uint32)

    # Here the new shape will be stored
    gpuShape = GPULightShape()

    gpuShape.size = len(shape.indices)
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)
    gpuShape.Ka = shape.Ka
    gpuShape.Ks = shape.Ks
    gpuShape.Kd = shape.Kd

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    gpuShape.texture = texture
    return gpuShape

#Funcion para dibujar una SceneGraph con iluminacion
def drawLightedNode(node, pipeline, transformName, parentTransform=tr.identity()):
    assert(isinstance(node, sg.SceneGraphNode))

    # Composing the transformations through this path
    newTransform = np.matmul(parentTransform, node.transform)

    # If the child node is a leaf, it should be a GPUShape.
    # Hence, it can be drawn with drawShape
    if len(node.childs) == 1 and isinstance(node.childs[0], es.GPUShape):
        leaf = node.childs[0]
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), leaf.Ka[0], leaf.Ka[1], leaf.Ka[2])
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), leaf.Kd[0], leaf.Kd[1], leaf.Kd[2])
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), leaf.Ks[0], leaf.Ks[1], leaf.Ks[2])
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, transformName), 1, GL_TRUE, newTransform)
        pipeline.drawShape(leaf)

    # If the child node is not a leaf, it MUST be a SceneGraphNode,
    # so this draw function is called recursively
    else:
        for child in node.childs:
            drawLightedNode(child, pipeline, transformName, newTransform)

#Pipeline normal(sin texturas e iluminacion) con transparencia
class AlphaTransformShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 130

            uniform mat4 transform;

            in vec3 position;
            in vec3 color;

            out vec3 newColor;

            void main()
            {
                gl_Position = transform * vec4(position, 1.0f);
                newColor = color;
            }
            """

        fragment_shader = """
            #version 130
            in vec3 newColor;

            out vec4 outColor;

            void main()
            {
                outColor = vec4(newColor, 0.6f);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))

    def drawShape(self, shape, mode=GL_TRIANGLES):
        assert isinstance(shape, es.GPUShape)

        # Binding the proper buffers
        glBindVertexArray(shape.vao)
        glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

        # 3d vertices + rgb color specification => 3*4 + 3*4 = 24 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        # Render the active element buffer with the active shader program
        glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)


#funcion para rotar un arreglo de vertices
def rotateShape(axis, angle, vertices):
    rot_vertices = []
    for v in range(len(vertices)//6):
        v_0 = vertices[6 * v + 0]
        v_1 = vertices[6 * v + 1]
        v_2 = vertices[6 * v + 2]
        r = vertices[6 * v + 3]
        g = vertices[6 * v + 4]
        b = vertices[6 * v + 5]
        rot_v = None
        if axis == "X":
            rot_v = tr.matmul([tr.rotationX(angle), np.array([v_0, v_1, v_2, 1]).T])
        elif axis == "Y":
            rot_v = tr.matmul([tr.rotationY(angle), np.array([v_0, v_1, v_2, 1]).T])
        elif axis == "Z":
            rot_v = tr.matmul([tr.rotationZ(angle), np.array([v_0, v_1, v_2, 1]).T])
        rot_vertex = np.array([rot_v[0], rot_v[1], rot_v[2]])/rot_v[3]
        rot_vertices += [rot_vertex[0], rot_vertex[1], rot_vertex[2], r, g, b]
    return rot_vertices

#Funcion para crear un rectangulo (sin texturas ni iluminacion) dados sus verrtices
def createRectangle(v1, v2, v3, v4, c1, c2, i_counter = 0):
    vertices = []
    indices = []

    '''
    Rectangle form
    v4 - - - - - - - - v3
    |                   |
    |                   |
    |                   |
    v1 - - - - - - - - v2
    '''

    vertices += [v1[0], v1[1], v1[2], c1[0], c1[0], c1[0]]
    vertices += [v2[0], v2[1], v2[2], c2[0], c2[0], c2[0]]
    vertices += [v3[0], v3[1], v3[2], c2[0], c2[0], c2[0]]
    vertices += [v4[0], v4[1], v4[2], c1[0], c1[0], c1[0]]
    indices += [i_counter + 0, i_counter + 1, i_counter + 2, i_counter + 2, i_counter + 3, i_counter + 0]
    return bs.Shape(vertices, indices)

#Funcion para crear un esfera (sin color ni texturas)
def createColorSphere(nTheta, nPhi, c1, c2, i_counter = 0):
    vertices = []
    indices = []
    theta_values = np.linspace(0, np.pi, nTheta + 1)
    phi_values = np.linspace(0, 2*np.pi, nPhi + 1)

    #vertice superior
    vertices += [0, 0, 0.5, c1[0], c1[1], c1[2]]
    #vertice inferior
    vertices += [0, 0,-0.5, c2[0], c2[1], c2[2]]
    vertex_counter = 2

    for dtheta in range(len(theta_values)):
        for dphi in range(len(phi_values)):
            #Se añaden las primeras caras
            color = c2*((dtheta+1)/len(theta_values)) + c1*(1-((dtheta+1)/len(theta_values)))
            r, g, b = color[0], color[1], color[2]
            if dtheta == 0:
                phi_0 = phi_values[dphi]
                theta_1 = theta_values[1]
                #se calcula la normal con el vector rho unitario de las coordenadas esfericas
                vertices += [0.5*np.sin(theta_1)*np.cos(phi_0), 0.5*np.sin(theta_1)*np.sin(phi_0), 0.5*np.cos(theta_1),
                             r, g, b]
                if dphi < len(phi_values)-1:
                    indices += [i_counter + 0,i_counter + dphi+2, i_counter + dphi + 3]

            #Se añaden las ultimas caras
            elif dtheta == len(theta_values)-1:
                if dphi < len(phi_values) - 1:
                    indices += [i_counter + 1, i_counter + vertex_counter - (nPhi + 1) + 1+ dphi, i_counter + vertex_counter - (nPhi + 1)+ dphi]

            else:
                phi_0 = phi_values[dphi]
                theta_1 = theta_values[dtheta]
                vertices += [0.5 * np.sin(theta_1) * np.cos(phi_0), 0.5 * np.sin(theta_1) * np.sin(phi_0),0.5 * np.cos(theta_1),
                             r, g, b]
                if dphi < len(phi_values)-1:
                    indices += [i_counter + vertex_counter + dphi, i_counter + vertex_counter + dphi +1, i_counter + vertex_counter - (nPhi + 1) + 1+ dphi]
                    indices += [i_counter + vertex_counter - (nPhi + 1) + 1+ dphi, i_counter + vertex_counter - (nPhi + 1)+ dphi, i_counter + vertex_counter + dphi]

        vertex_counter += (nPhi + 1)

    return bs.Shape(vertices, indices)
