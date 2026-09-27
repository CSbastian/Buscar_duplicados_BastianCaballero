# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# Fecha: 23 de agosto de 2025
# Alumno: [nombre]

# Este archivo es usado por tarea1-parte1.py y tarea1-parte2.py
# Permite tener funciones compartidas entre ambos programas

# Se puede modificar este archivo para agregar más funciones (si es necesario)
# Incluir este archivo en la entrega de la tarea si fue modificado.

import os
import pickle
import numpy
import cv2


# Retorna todos los archivos que terminan con el parametro extension
# ejemplo: listar_archivos_con_extension(dir, ".jpg") retorna los archivos en dir cuyo nombre termina con .jpg
def listar_archivos_con_extension(carpeta, extension):
    lista = []
    for archivo in os.listdir(carpeta):
        # los que terminan con la extension se agregan a la lista de nombres
        if archivo.endswith(extension):
            lista.append(archivo)
    lista.sort()
    return lista


# escribe el objeto de python en un archivo binario
def guardar_objeto(objeto, carpeta, nombre_archivo):
    if carpeta == "" or carpeta == "." or carpeta is None:
        archivo = nombre_archivo
    else:
        archivo = os.path.join(carpeta, nombre_archivo)
        # asegura que la carpeta exista
        os.makedirs(carpeta, exist_ok=True)
    # usa la librería pickle para escribir el objeto en un archivo binario
    # ver https://docs.python.org/3/library/pickle.html
    with open(archivo, "wb") as handle:
        pickle.dump(objeto, handle, protocol=pickle.HIGHEST_PROTOCOL)


# reconstruye el objeto de python que está guardado en un archivo
def leer_objeto(carpeta, nombre_archivo):
    if carpeta == "" or carpeta == "." or carpeta is None:
        archivo = nombre_archivo
    else:
        archivo = os.path.join(carpeta, nombre_archivo)
    with open(archivo, "rb") as handle:
        objeto = pickle.load(handle)
    return objeto


# Recibe una lista de listas y lo escribe en un archivo separado por \t
# Por ejemplo:
# listas = [
#           ["dato1a", "dato1b", "dato1c"],
#           ["dato2a", "dato2b", "dato2c"],
#           ["dato3a", "dato3b", "dato3c"] ]
# al llamar:
#   escribir_lista_de_columnas_en_archivo(listas, "archivo.txt")
# escribe un archivo de texto con:
# dato1a  dato1b   dato1c
# dato2a  dato2b   dato3c
# dato2a  dato2b   dato3c
def escribir_lista_de_columnas_en_archivo(lista_con_columnas, archivo_texto_salida):
    with open(archivo_texto_salida, "w") as handle:
        for columnas in lista_con_columnas:
            textos = []
            for col in columnas:
                textos.append(str(col))
            texto = "\t".join(textos)
            print(texto, file=handle)





##########Funciones Aux#####################
def crear_matriz_descriptores(imagenes, dir_input_imagenes_R, funcion_descriptor):
    matriz_descriptores = None
    num_fila = 0
    for nombre_imagen in imagenes:
        descriptor_imagen = funcion_descriptor(nombre_imagen, dir_input_imagenes_R)
        # crear la matriz de descriptores (numero imagenes x largo_descriptor)
        if matriz_descriptores is None:
            matriz_descriptores = numpy.zeros((len(imagenes), len(descriptor_imagen)), numpy.float32)
        # copiar descriptor a una fila de la matriz de descriptores
        matriz_descriptores[num_fila] = descriptor_imagen
        num_fila += 1
    # 3-escribir en dir_output_descriptores_R los d
    return matriz_descriptores


def vector_de_intensidades_omd(nombre_imagen, imagenes_dir):
    archivo_imagen = imagenes_dir + "/" + nombre_imagen
    imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise Exception("no puedo abrir: " + archivo_imagen)
    # se puede cambiar a otra interpolacion, como cv2.INTER_CUBIC
    imagen_reducida = cv2.resize(imagen, (5, 5), interpolation=cv2.INTER_AREA)
    # flatten convierte una matriz de nxm en un array de largo nxm
    descriptor_imagen = imagen_reducida.flatten()
    # la posicion si se ordenan
    posiciones = numpy.argsort(descriptor_imagen)
    # reemplazar el valor gris por su  posicion
    for i in range(len(posiciones)):
        descriptor_imagen[posiciones[i]] = i
    return descriptor_imagen


def histograma_por_zona(imagen, imagen_hists):
    # divisiones
    num_zonas_x = 5
    num_zonas_y = 5
    num_bins_por_zona = 32
    # procesar cada zona
    descriptor = []
    for j in range(num_zonas_y):
        desde_y = int(imagen.shape[0] / num_zonas_y * j)
        hasta_y = int(imagen.shape[0] / num_zonas_y * (j + 1))
        for i in range(num_zonas_x):
            desde_x = int(imagen.shape[1] / num_zonas_x * i)
            hasta_x = int(imagen.shape[1] / num_zonas_x * (i + 1))
            # recortar zona de la imagen
            zona = imagen[desde_y: hasta_y, desde_x: hasta_x]
            # histograma de los pixeles de la zona
            histograma, limites = numpy.histogram(zona, bins=num_bins_por_zona, range=(0, 255))
            # normalizar histograma (bins suman 1)
            histograma = histograma / numpy.sum(histograma)
            # agregar descriptor de la zona al descriptor global
            descriptor.extend(histograma)
    return descriptor


def histogramas_por_zonas(nombre_imagen, imagenes_dir):
    archivo_imagen = imagenes_dir + "/" + nombre_imagen
    imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise Exception("no puedo abrir: " + archivo_imagen)
    # ecualizacion
    imagen = cv2.equalizeHist(imagen)
    descriptor = histograma_por_zona(imagen, None)
    return descriptor

def histogramas_por_zonas_con_flip(nombre_imagen, imagenes_dir):
    archivo_imagen = os.path.join(imagenes_dir, nombre_imagen)
    imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise Exception("no puedo abrir: " + archivo_imagen)

    # Aplicar flip horizontal
    imagen = cv2.flip(imagen, 1)

    # Ecualización de histograma
    imagen = cv2.equalizeHist(imagen)
    # Calcular histogramas por zonas en la imagen con flip
    descriptor = histograma_por_zona(imagen, None)
    return descriptor


#-----------------------------
# hace flip a la imagen q y genera el descriptor OMD
def omd_con_flip(nombre_imagen, imagenes_dir):

    archivo_imagen = imagenes_dir + "/" + nombre_imagen
    imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise Exception("no puedo abrir: " + archivo_imagen)

    # Aplicar flip
    imagen = cv2.flip(imagen, 1)

    # Calculo omd
    imagen_reducida = cv2.resize(imagen, (5, 5), interpolation=cv2.INTER_AREA)
    descriptor_imagen = imagen_reducida.flatten()
    posiciones = numpy.argsort(descriptor_imagen)
    for i in range(len(posiciones)):
        descriptor_imagen[posiciones[i]] = i
    return descriptor_imagen