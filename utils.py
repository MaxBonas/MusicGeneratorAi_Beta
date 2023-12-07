import numpy as np

# Función para guardar secuencias en un archivo .npy
def guardar_secuencia(secuencia, ruta_archivo):
    np.save(ruta_archivo, secuencia)
    print(f"Secuencia guardada en {ruta_archivo}")

# Función para cargar secuencias de un archivo .npy
def cargar_secuencia(ruta_archivo):
    secuencia = np.load(ruta_archivo)
    print(f"Secuencia cargada de {ruta_archivo}")
    return secuencia