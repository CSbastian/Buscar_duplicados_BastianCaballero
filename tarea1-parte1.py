# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# Fecha: 23 de agosto de 2025
# Alumno: [nombre]

import sys
import os
import t1_util
import numpy
import cv2


#descriptor:





def tarea1_parte1(dir_input_imagenes_R, dir_output_descriptores_R):
    if not os.path.isdir(dir_input_imagenes_R):
        print("ERROR: no existe directorio {}".format(dir_input_imagenes_R))
        sys.exit(1)
    elif os.path.exists(dir_output_descriptores_R):
        print("ERROR: ya existe directorio {}".format(dir_output_descriptores_R))
        sys.exit(1)
    # Implementar la fase offline

    # 1-leer imágenes en dir_input_imagenes
    # puede servir la funcion t1_util.listar_archivos_con_extension() que está definida en t1_util.py
    imagenes = t1_util.listar_archivos_con_extension(dir_input_imagenes_R, ".jpg")

    # 2-calcular descriptores de imágenes
    # ver codigo de ejemplo publicado en el curso
    matriz_descriptores = t1_util.crear_matriz_descriptores(imagenes, dir_input_imagenes_R, t1_util.histogramas_por_zonas)

    #3-guardar descriptores
    # puede servir la funcion t1_util.guardar_objeto() que está definida en t1_util.py
    t1_util.guardar_objeto(matriz_descriptores, dir_output_descriptores_R, "descriptores.pkl")


# inicio de la tarea
if len(sys.argv) < 3:
    print("Uso: {}  dir_input_imagenes_R  dir_output_descriptores_R".format(sys.argv[0]))
    sys.exit(1)

# lee los parametros de entrada
dir_input_imagenes_R = sys.argv[1]
dir_output_descriptores_R = sys.argv[2]

# ejecuta la tarea
tarea1_parte1(dir_input_imagenes_R, dir_output_descriptores_R)
