from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

BATCH_SIZE = 32
NUM_EPOCHS = 15

print('Loading data...')

def load_training_data():
    
    return X_train, y_train, X_test, y_test

def build_model():
    print('Pad sequences (samples x time)')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, input_dim=9))  # try using a GRU instead, for fun
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    print(X_train.shape)
    print(y_train.shape)

def train(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
              validation_data=(X_test, y_test))

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=BATCH_SIZE)
    print('Test score:', score)
    print('Test accuracy:', acc)

def train(model, X_train, y_train, X_test, y_test):

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_training_data()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test)