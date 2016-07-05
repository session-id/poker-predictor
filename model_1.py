from __future__ import print_function

import os
import numpy as np
import logging
import argparse

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ProgbarLogger, Callback

BATCH_SIZE = 64
NUM_EPOCHS = 5
INPUT_DIM = 14
INPUT_LENGTH = 20
OUTPUT_DIM = 3
INTER_DIM = (20, 10)
TRAINING_DATA_DIR = "training_data"
USE_ONE_TRAINING_FILE = False
TRAIN_DATA_RATIO = 0.75 # Amount of total data to use for training

np.random.seed(1337)  # for reproducibility

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

class PrintLoss(Callback):
    def __init__(self, pad_ratio):
        self.pad_ratio = pad_ratio

    def on_epoch_end(self, epoch, logs={}):
        logging.info('True Loss: {0}'.format(logs.get('loss') * self.pad_ratio))

# NUM_SAMPLES / NUM_NON_PADDED_SAMPLES
def pad_ratio(y):
    return float(y.shape[0] * y.shape[1]) / np.sum(y, (0,1,2))

def load_training_data():
    logging.info('Loading data...')

    # Only for testing
    if USE_ONE_TRAINING_FILE:
        filename = "training_data/training_0.npz";
        with open(filename) as f:
            data = np.load(f)
            X_train = data["input"]
            y_train = data["output"]
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

    logging.info("Shape of X_train: ")
    logging.info(X_train.shape)
    logging.info("Shape of y_train")
    logging.info(y_train.shape)

    train_test_sep_idx = int(X_train.shape[0] * TRAIN_DATA_RATIO)
    X_test = X_train[train_test_sep_idx:]
    y_test = y_train[train_test_sep_idx:]
    X_train = X_train[:train_test_sep_idx]
    y_train = y_train[:train_test_sep_idx]

    return X_train, y_train, X_test, y_test

def build_model(processor):
    logging.info('Build model...')
    model = Sequential()

    model.add(LSTM(INTER_DIM[0], return_sequences=True, dropout_W=0.2, dropout_U=0.2,
                   input_length=INPUT_LENGTH, input_dim=INPUT_DIM, consume_less=processor))
    model.add(LSTM(INTER_DIM[1], return_sequences=True, dropout_W=0.2, dropout_U=0.2,
                   input_length=INPUT_LENGTH, input_dim=INTER_DIM[0], consume_less=processor))
    model.add(TimeDistributed(Dense(OUTPUT_DIM)))

    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)    
    return model

def train(model, X_train, y_train, X_test, y_test, start_weights_file=None):
    logging.info('Train...')
    save_weights = ModelCheckpoint('weights.{epoch:02d}.hdf5')
    logger = ProgbarLogger()
    printloss = PrintLoss(pad_ratio(y_test))

    if start_weights_file:
        model.load_weights(start_weights_file)

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
              validation_data=(X_test, y_test), callbacks=[save_weights, logger, printloss])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=BATCH_SIZE)
    # logging.info('Test score: {0}'.format(score))
    # logging.info('Test accuracy: {0}'.format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poker Predictor.')
    parser.add_argument('--file', type=str, help='sum the integers (default: find the max)')
    parser.add_argument('--gpu', const='gpu', default='cpu', nargs="?", 
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_training_data()
    logging.info("Padding ratio (multiply loss by this): ")
    logging.info(pad_ratio(y_test))

    model = build_model(args.gpu)
    train(model, X_train, y_train, X_test, y_test, start_weights_file=args.file)
