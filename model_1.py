from __future__ import print_function

import os
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils.visualize_util import plot

BATCH_SIZE = 32
NUM_EPOCHS = 15
INPUT_DIM = 8
OUTPUT_DIM = 4
TRAINING_DATA_DIR = "training_data"
USE_ONE_TRAINING_FILE = False

def load_training_data():
    print('Loading data...')

    # Only for testing
    if USE_ONE_TRAINING_FILE:
        filename = "training_data/training_0.npz";
        with open(filename) as f:
            data = np.load(f)
            X_train = data["input"]
            y_train = data["output"]
        X_test = X_train
        y_test = y_train
    else:
        Xs = []
        ys = []
        for filename in os.listdir(TRAINING_DATA_DIR):
            full_name = TRAINING_DATA_DIR + "/" + filename;
            with open(full_name) as f:
                data = np.load(f)
                Xs.append(data["input"])
                ys.append(data["output"])
        X_train = np.concatenate(tuple(Xs))
        y_train = np.concatenate(tuple(ys))
        X_test = X_train
        y_test = y_train

    print("Shape of X_train: ")
    print(X_train.shape)
    print("Shape of y_train")
    print(y_train.shape)
    
    return X_train, y_train, X_test, y_test

def build_model():
    print('Build model...')
    model = Sequential()
    model.add(LSTM(OUTPUT_DIM, dropout_W=0.2, dropout_U=0.2, input_dim=INPUT_DIM))  # try using a GRU instead, for fun
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)    
    return model

def train(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
              validation_data=(X_test, y_test))

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=BATCH_SIZE)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_training_data()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test)
