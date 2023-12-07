#model.py

import logging
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from config import (
    NUM_NOTAS, NUM_UNITS_LSTM, DROPOUT_RATE, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT,
    FILEPATH_CHECKPOINT, PATIENCE_EARLY_STOPPING, PATIENCE_REDUCE_LR, FACTOR_REDUCE_LR, MIN_LR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cargar_datos_entrenamiento(ruta):
    X_train = np.load(ruta + '/X_train.npy')
    y_train = np.load(ruta + '/y_train.npy')
    logging.info(f"Datos de entrenamiento cargados desde {ruta}")
    return X_train, y_train

def crear_modelo(input_shape):
    num_output_units = NUM_NOTAS + 2  # Notas en codificación one-hot + duración y tiempo

    model = Sequential()
    model.add(LSTM(NUM_UNITS_LSTM, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(NUM_UNITS_LSTM))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(num_output_units, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def entrenar_modelo(model, X_train, y_train):
    callbacks_list = [
        ModelCheckpoint(FILEPATH_CHECKPOINT, monitor='loss', verbose=1, save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY_STOPPING, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=FACTOR_REDUCE_LR, patience=PATIENCE_REDUCE_LR, verbose=1, min_lr=MIN_LR)
    ]

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=1)

    logging.info("Entrenamiento completado")
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        logging.info(f"Epoch {epoch+1}: Loss = {loss}, Val Loss = {val_loss}")

    return model, history

if __name__ == "__main__":
    ruta_datos = 'ruta/a/tus/datos'
    X_train, y_train = cargar_datos_entrenamiento(ruta_datos)

    input_shape = (X_train.shape[1], X_train.shape[2])
    modelo = crear_modelo(input_shape)
    modelo, history = entrenar_modelo(modelo, X_train, y_train)


