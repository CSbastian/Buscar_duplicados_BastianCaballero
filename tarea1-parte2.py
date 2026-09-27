# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# Fecha: 23 de agosto de 2025
# Alumno: [nombre]

import sys
import os

import scipy
import t1_util
import numpy
import cv2


#descriptor:
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


def tarea1_parte2(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados):
    if not os.path.isdir(dir_input_imagenes_Q):
        print("ERROR: no existe directorio {}".format(dir_input_imagenes_Q))
        sys.exit(1)
    elif not os.path.isdir(dir_input_descriptores_R):
        print("ERROR: no existe directorio {} (¿terminó bien tarea1-parte1.py?)".format(dir_input_descriptores_R))
        sys.exit(1)
    elif os.path.exists(file_output_resultados):
        print("ERROR: ya existe archivo {}".format(file_output_resultados))
        sys.exit(1)
    # Implementar la fase online

    # 1-calcular descriptores de Q para imágenes en dir_input_imagenes_Q
    # arreglo de nombres de imágenes en Q
    imagenes_q = t1_util.listar_archivos_con_extension(dir_input_imagenes_Q, ".jpg")
    # calcular matriz de descriptores de Q usando omd
    descriptores_q = t1_util.crear_matriz_descriptores(imagenes_q, dir_input_imagenes_Q, t1_util.histogramas_por_zonas)

    # calcular descriptores de Q usando OMD con flip
    descriptores_q_flip = t1_util.crear_matriz_descriptores(imagenes_q, dir_input_imagenes_Q,
                                                         t1_util.histogramas_por_zonas_con_flip)
    # 2-leer descriptores de R guardados en dir_input_descriptores_R
    descriptores_r = t1_util.leer_objeto(dir_input_descriptores_R, "descriptores.pkl")
    # 3-para cada descriptor q localizar el mas cercano en R
    #arreglo con los resultados consta de nombre_imagen_q, nombre_imagen_r, distancia
    resultados = []
    matriz_distancias = scipy.spatial.distance.cdist(descriptores_q, descriptores_r, metric='cityblock')
    matriz_distancias_flip = scipy.spatial.distance.cdist(descriptores_q_flip, descriptores_r, metric='cityblock')
    for i in range(matriz_distancias.shape[0]):  # por cada imagen q

        #busco las distancias minimas para cada matriz
        indice_r_mas_cercano = numpy.argmin(matriz_distancias[i, :])
        distancia_minima = matriz_distancias[i, indice_r_mas_cercano]
        indice_r_mas_cercano_flip = numpy.argmin(matriz_distancias_flip[i, :])
        distancia_minima_flip = matriz_distancias_flip[i, indice_r_mas_cercano_flip]

        #chequeo cual es la minima entre las dos
        if distancia_minima_flip < distancia_minima:
            indice_r = indice_r_mas_cercano_flip
            distancia_minima = distancia_minima_flip
        else:
            indice_r = indice_r_mas_cercano
            distancia_minima = distancia_minima
        # buscamos las imagenes coincidentes
        nombre_imagen_q = imagenes_q[i]
        nombre_imagen_r = "r{:04d}.jpg".format(indice_r + 1)
        # agregamos el resultado
        resultados.append([nombre_imagen_q, nombre_imagen_r, distancia_minima])
    # 4-escribir en el archivo file_output_resultados un archivo con tres columnas separado por \t:
    # columna 1: imagen_q
    # columna 2: imagen_r
    # columna 3: distancia
    t1_util.escribir_lista_de_columnas_en_archivo(resultados, file_output_resultados)


# inicio de la tarea
if len(sys.argv) < 4:
    print("Uso: {}  dir_input_imagenes_Q  dir_input_descriptores_R  file_output_resultados".format(sys.argv[0]))
    sys.exit(1)

# lee los parametros de entrada
dir_input_imagenes_Q = sys.argv[1]
dir_input_descriptores_R = sys.argv[2]
file_output_resultados = sys.argv[3]

# ejecuta la tarea
tarea1_parte2(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados)
