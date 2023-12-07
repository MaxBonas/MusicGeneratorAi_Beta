# config.py

# Configuraciones generales de MIDI
NUM_NOTAS = 128
MAX_DURACION = 480 * 4  # Ejemplo: 4 compases con tiempo de cuarto de nota = 480

# Parámetros de modelo de aprendizaje automático
NUM_UNITS_LSTM = 256
DROPOUT_RATE = 0.3
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# Rutas de archivos y directorios
RUTA_DATOS_ENTRENAMIENTO = 'ruta/a/tus/datos'
RUTA_MODELO = 'modelo_entrenado.keras'

# Parámetros de entrenamiento y callback
FILEPATH_CHECKPOINT = "weights-improvement-{epoch:02d}-{loss:.4f}.keras"
PATIENCE_EARLY_STOPPING = 10
PATIENCE_REDUCE_LR = 5
FACTOR_REDUCE_LR = 0.2
MIN_LR = 0.001

# Otras configuraciones que puedas necesitar
