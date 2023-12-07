# data_loader.py

import mido
import numpy as np
import os
import logging
from tqdm import tqdm
from config import NUM_NOTAS, MAX_DURACION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cargar_archivo_midi(ruta_archivo):
    try:
        midi_data = mido.MidiFile(ruta_archivo)
        logging.info(f"Archivo MIDI cargado: {ruta_archivo}")
        return midi_data
    except Exception as e:
        logging.error(f"Error al cargar el archivo MIDI {ruta_archivo}: {e}")
        return None

def extraer_notas_y_ritmo(midi_data):
    notas, duraciones, tiempos = [], [], []
    tiempo_actual = 0
    estado_notas = {}

    for track in midi_data.tracks:
        for msg in track:
            tiempo_actual += msg.time
            if msg.type == 'note_on':
                estado_notas[msg.note] = tiempo_actual
            elif msg.type == 'note_off':
                if msg.note in estado_notas:
                    inicio_nota = estado_notas.pop(msg.note)
                    duracion_nota = tiempo_actual - inicio_nota
                    notas.append(msg.note)
                    duraciones.append(duracion_nota)
                    tiempos.append(inicio_nota)
    return notas, duraciones, tiempos

def cargar_y_procesar_archivos_midi(directorio_principal):
    todas_las_notas, todas_las_duraciones, todos_los_tiempos = [], [], []

    for nombre_subcarpeta in os.listdir(directorio_principal):
        ruta_subcarpeta = os.path.join(directorio_principal, nombre_subcarpeta)
        if os.path.isdir(ruta_subcarpeta):
            for archivo in os.listdir(ruta_subcarpeta):
                ruta_completa = os.path.join(ruta_subcarpeta, archivo)
                if archivo.endswith('.midi') or archivo.endswith('.mid'):
                    midi_data = cargar_archivo_midi(ruta_completa)
                    if midi_data:
                        notas, duraciones, tiempos = extraer_notas_y_ritmo(midi_data)
                        todas_las_notas.extend(notas)
                        todas_las_duraciones.extend(duraciones)
                        todos_los_tiempos.extend(tiempos)
    return todas_las_notas, todas_las_duraciones, todos_los_tiempos

def codificar(nota, duracion, tiempo):
    # Codificación one-hot para la nota
    nota_codificada = np.zeros(NUM_NOTAS)
    nota_codificada[nota] = 1

    # Normalización de la duración y tiempo
    duracion_normalizada = duracion / MAX_DURACION
    tiempo_normalizado = tiempo / MAX_DURACION

    return np.concatenate([nota_codificada, [duracion_normalizada, tiempo_normalizado]])


def preparar_datos_para_modelo(notas, duraciones, tiempos, longitud_secuencia=30):
    X, y = [], []

    for i in range(len(notas) - longitud_secuencia):
        secuencia_entrada = []
        secuencia_salida = notas[i + longitud_secuencia]

        for j in range(i, i + longitud_secuencia):
            entrada = codificar(notas[j], duraciones[j], tiempos[j])
            secuencia_entrada.append(entrada)

        X.append(secuencia_entrada)
        y.append(secuencia_salida)

    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    directorio_principal = 'data/maestro-v1.0.0'
    notas, duraciones, tiempos = cargar_y_procesar_archivos_midi(directorio_principal)
    logging.info(f"Total de notas extraídas: {len(notas)}")
    logging.info(f"Total de duraciones: {len(duraciones)}")
    logging.info(f"Total de tiempos: {len(tiempos)}")
