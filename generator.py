# generator.py

import numpy as np
from keras.models import load_model
import mido
import logging
import random
from config import NUM_NOTAS, MAX_DURACION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cargar_modelo(filepath):
    return load_model(filepath)

def generar_musica(model, seed_sequence, generation_length=500):
    generated_sequence = []
    current_sequence = seed_sequence

    for _ in range(generation_length):
        predicted_step = model.predict(current_sequence)
        predicted_note = np.argmax(predicted_step[:, :NUM_NOTAS], axis=-1)
        predicted_duration = np.argmax(predicted_step[:, NUM_NOTAS:], axis=-1)
        generated_sequence.append((predicted_note[0], predicted_duration[0]))

        # Preparar la siguiente secuencia de entrada
        next_step = np.zeros((1, 1, NUM_NOTAS + 2))
        next_step[0, 0, predicted_note[0]] = 1
        next_step[0, 0, NUM_NOTAS:] = predicted_step[0, NUM_NOTAS:]
        current_sequence = np.concatenate((current_sequence[:, 1:, :], next_step), axis=1)

    return generated_sequence

def secuencia_a_midi(secuencia, archivo_salida='musica_generada.mid'):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tiempo_acumulado = 0
    for nota, duracion in secuencia:
        if nota > 0:
            track.append(mido.Message('note_on', note=nota, velocity=64, time=tiempo_acumulado))
            track.append(mido.Message('note_off', note=nota, velocity=64, time=duracion))
            tiempo_acumulado = 0
        else:
            tiempo_acumulado += duracion

    mid.save(archivo_salida)
    logging.info(f"Archivo MIDI generado guardado como {archivo_salida}")

def obtener_secuencia_inicial(metodo, datos=None, longitud_secuencia=30, secuencia_especifica=None):
    if metodo == "aleatoria":
        if datos is None:
            raise ValueError("Se requieren 'datos' para el método 'aleatoria'")
        inicio = random.randint(0, len(datos) - longitud_secuencia)
        return datos[inicio:inicio + longitud_secuencia].reshape(1, longitud_secuencia, -1)
    elif metodo == "especifica":
        if secuencia_especifica is None:
            raise ValueError("Se requiere una 'secuencia_especifica' para el método 'especifica'")
        return np.array(secuencia_especifica).reshape(1, longitud_secuencia, -1)
    elif metodo == "completamente_aleatoria":
        secuencia = []
        for _ in range(longitud_secuencia):
            nota_aleatoria = random.randint(0, NUM_NOTAS-1)
            duracion_aleatoria = random.randint(1, MAX_DURACION)
            secuencia.append((nota_aleatoria, duracion_aleatoria))
        return np.array(secuencia).reshape(1, longitud_secuencia, -1)
    else:
        raise ValueError("Método no reconocido. Elija 'aleatoria', 'especifica' o 'completamente_aleatoria'.")


if __name__ == "__main__":
    model_filepath = 'modelo_entrenado.keras'
    model = cargar_modelo(model_filepath)

    # Configurar la secuencia inicial según el método deseado
    metodo_secuencia = "aleatoria"  # Cambiar a "especifica" o "completamente_aleatoria" según sea necesario
    secuencia_especifica = [(60, 480), (62, 480), (64, 480)] if metodo_secuencia == "especifica" else None
    datos = ... if metodo_secuencia == "aleatoria" else None

    seed_sequence = obtener_secuencia_inicial(metodo_secuencia, datos=datos, secuencia_especifica=secuencia_especifica)

    generated_music = generar_musica(model, seed_sequence)
    secuencia_a_midi(generated_music)
