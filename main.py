# main.py

import logging
from data_loader import cargar_y_procesar_archivos_midi, preparar_datos_para_modelo
from model import crear_modelo, entrenar_modelo, cargar_datos_entrenamiento
from generator import cargar_modelo, generar_musica, secuencia_a_midi, obtener_secuencia_inicial
from config import RUTA_DATOS_ENTRENAMIENTO, RUTA_MODELO

class Main:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def ejecutar_proceso_completo(self):
        # Cargar y procesar datos MIDI
        directorio_principal = 'data/maestro-v1.0.0'
        notas, duraciones, tiempos = cargar_y_procesar_archivos_midi(directorio_principal)
        self.logger.info(f"Total de notas extraídas: {len(notas)}")

        # Preparar datos para el modelo
        X_train, y_train = preparar_datos_para_modelo(notas, duraciones, tiempos)

        # Cargar datos de entrenamiento (si es necesario)
        X_train, y_train = cargar_datos_entrenamiento(RUTA_DATOS_ENTRENAMIENTO)

        # Crear y entrenar el modelo
        input_shape = (X_train.shape[1], X_train.shape[2])
        modelo = crear_modelo(input_shape)
        modelo, history = entrenar_modelo(modelo, X_train, y_train)

        # Generar música con el modelo entrenado
        metodo_secuencia = "aleatoria"  # O "especifica" o "completamente_aleatoria"
        secuencia_especifica = [(60, 480), (62, 480), (64, 480)] if metodo_secuencia == "especifica" else None
        seed_sequence = obtener_secuencia_inicial(metodo_secuencia, datos=X_train, secuencia_especifica=secuencia_especifica)

        generated_music = generar_musica(modelo, seed_sequence)
        secuencia_a_midi(generated_music, 'musica_generada.mid')

        self.logger.info("Proceso completo finalizado.")

if __name__ == "__main__":
    main = Main()
    main.ejecutar_proceso_completo()
