# Buscador de Imágenes Duplicadas

Este proyecto implementa un sistema de recuperación de imágenes por similitud. Su objetivo principal es encontrar copias y versiones modificadas de imágenes dadas un conjunto de consulta (Query) frente a un conjunto de referencia (Dataset).

## Características y Técnicas Usadas

- **Descriptores Globales**: Se emplea el cálculo de descriptores utilizando histogramas de intensidad en escala de grises distribuidos por zonas. Esto permite capturar información espacial básica combinada con los niveles de iluminación.
- **Aumento de Datos (Data Augmentation) para Robustez**: Se realiza un proceso de doble pasada donde las imágenes de consulta se procesan tanto en su versión original como aplicándoles un reflejo horizontal (FLIP). Esto incrementa drásticamente la capacidad del algoritmo para reconocer imágenes espejadas.
- **Métrica de Distancia**: Se usa la distancia *Cityblock* (Manhattan) implementada a través de `scipy.spatial.distance.cdist` para determinar la proximidad entre descriptores y encontrar coincidencias más confiables y veloces.
- **Fase Offline y Online**: 
  - **Offline (`tarea1-parte1.py`)**: Precalcula y guarda los descriptores de la base de datos de referencia (R) en formato `.pkl`, optimizando los tiempos de búsqueda futuros.
  - **Online (`tarea1-parte2.py`)**: Computa dinámicamente los descriptores de las imágenes de consulta (Q) y busca el vector más cercano en R.

## Requisitos

El proyecto está programado en Python y necesita las siguientes librerías:

```bash
pip install numpy scipy opencv-python
```

## Uso

### Fase 1: Creación de Descriptores (Offline)

Precalcula los descriptores del dataset base de imágenes.

```bash
python tarea1-parte1.py <carpeta_dataset_R> <carpeta_descriptores_R>
```

### Fase 2: Búsqueda (Online)

Busca las imágenes de consulta dentro del dataset indexado y emite los resultados.

```bash
python tarea1-parte2.py <carpeta_consultas_Q> <carpeta_descriptores_R> <archivo_resultados_txt>
```

El archivo final de resultados contendrá 3 columnas tabuladas correspondientes a: `[Nombre_imagen_Q] [Nombre_imagen_R] [Distancia_calculada]`.

## Resultados

En los benchmarks originales, el sistema alcanzó alrededor de un **~70% de aciertos** (F1-score de 0.686) lidiando con imágenes modificadas mediante recortes (CROP), cambios de gamma (GAMMA), marcos (FRAMED) y variaciones de perspectiva, logrando excelentes métricas para rotaciones horizontales y cambios de calidad (QUALITY).