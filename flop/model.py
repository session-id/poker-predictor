# from __future__ import print_function

import os
import numpy as np
import logging
import argparse

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM, merge, Input
from keras.callbacks import ModelCheckpoint, ProgbarLogger, Callback

BATCH_SIZE = 500
NUM_EPOCHS = 5
INPUT_DIM = 16
FLOP_DIM = 42
OUTPUT_DIM = 3
INPUT_LENGTH = 20
INTER_DIM = (30, 10)
FLOP_INTER_DIM = (30, 20, 10)
TRAINING_DATA_DIR = "training_data"
TRAIN_DATA_RATIO = 0.75 # Amount of total data to use for training

SINGLE_TRAINING_FILENAME =  "training_data/training_2.npz"
USE_ONE_TRAINING_FILE = False
MAX_TRAINING_FILES = 1000

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
        with open(SINGLE_TRAINING_FILENAME) as f:
            data = np.load(f)
            X_train = data["input"]
            y_train = data["output"]
            flops_train = data["board"]
    else:
        Xs = []
        ys = []
        flops = []
        files = os.listdir(TRAINING_DATA_DIR)
        np.random.shuffle(files)
        for i, filename in enumerate(files):
            if i > MAX_TRAINING_FILES:
                break
            full_name = TRAINING_DATA_DIR + "/" + filename
            with open(full_name) as f:
                data = np.load(f)
                if len(data["input"].shape) == 3 and len(data["output"].shape) == 3:
                    Xs.append(data["input"])
                    ys.append(data["output"])
                    flops.append(data["board"])
            logging.info(str(i))
        X_train = np.concatenate(tuple(Xs))
        y_train = np.concatenate(tuple(ys))
        flops_train = np.concatenate(tuple(flops))

    # Expand flops_train
    flops_train = np.tile(np.expand_dims(flops_train, 1), (1, INPUT_LENGTH, 1))

    logging.info("Shape of X_train: ")
    logging.info(X_train.shape)
    logging.info("Shape of y_train")
    logging.info(y_train.shape)
    logging.info("Shape of flops_train")
    logging.info(flops_train.shape)

    # Randomize hands
    rand_perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[rand_perm]
    y_train = y_train[rand_perm]
    flops_train = flops_train[rand_perm]

    train_test_sep_idx = int(X_train.shape[0] * TRAIN_DATA_RATIO)
    X_test = X_train[train_test_sep_idx:]
    y_test = y_train[train_test_sep_idx:]
    X_train = X_train[:train_test_sep_idx]
    y_train = y_train[:train_test_sep_idx]
    flops_test = flops_train[train_test_sep_idx:]
    flops_train = flops_train[:train_test_sep_idx]

    return X_train, flops_train, y_train, X_test, flops_test, y_test

def build_model(processor):
    logging.info('Build model...')

    action_input = Input(shape=(INPUT_LENGTH, INPUT_DIM))
    actual_flop_input = Input(shape=(INPUT_LENGTH, FLOP_DIM))
    flop_input = actual_flop_input

    # 2 dense layers to encode flop
    for dim in FLOP_INTER_DIM:
        flop_input = TimeDistributed(Dense(dim))(flop_input)

    seq = merge([action_input, flop_input], mode='concat', concat_axis=2)
    
    for dim in INTER_DIM:
        seq = LSTM(dim, return_sequences=True, dropout_W=0.2, dropout_U=0.2,
                   consume_less=processor)(seq)
    seq = TimeDistributed(Dense(OUTPUT_DIM))(seq)
    probs = Activation('softmax')(seq)

    model = Model(input=[action_input, actual_flop_input], output=probs)

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
    parser.add_argument('--file', type=str, help='Start training weights.')
    parser.add_argument('--gpu', const='gpu', default='cpu', nargs="?", 
                        help='sum the integers (default: find the max)')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--max-training-files', type=int,
                        help='Maximum number of training files to use.')

    args = parser.parse_args()

    if args.batch_size:
        BATCH_SIZE = args.batch_size

    if args.max_training_files:
        MAX_TRAINING_FILES = args.max_training_files

    X_train, flops_train, y_train, X_test, flops_test, y_test = load_training_data()
    logging.info("Padding ratio (multiply loss by this): ")
    logging.info(pad_ratio(y_test))

    model = build_model(args.gpu)
    train(model, [X_train, flops_train], y_train, [X_test, flops_test], y_test, start_weights_file=args.file)
