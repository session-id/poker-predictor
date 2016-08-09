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
INPUT_DIM = 17
INPUT_LENGTH = 20
OUTPUT_DIM = 3
INTER_DIM = (20, 10, 5, 4)
TRAINING_DATA_DIR = "../adaptive/training_data"
USE_ONE_TRAINING_FILE = False
TRAIN_DATA_RATIO = 0.75 # Amount of total data to use for training
CLUSTER_FILENAME = "../adaptive/vpip_pfr_clusters.csv"
MAX_TRAINING_FILES = 1000000

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
        filename = "../adaptive/training_data/00d5UcT1hvMMwvmgryyCuA.npz"
        with open(filename) as f:
            data = np.load(f)
            X_train = data["input"]
            y_train = data["output"]
    else:
        Xs = []
        ys = []
        files = os.listdir(TRAINING_DATA_DIR)
        np.random.shuffle(files)
        for i, filename in enumerate(files):
            if i >= MAX_TRAINING_FILES:
                break
            full_name = TRAINING_DATA_DIR + "/" + filename
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

    # Randomize hands
    rand_perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[rand_perm]
    y_train = y_train[rand_perm]

    train_test_sep_idx = int(X_train.shape[0] * TRAIN_DATA_RATIO)
    X_test = X_train[train_test_sep_idx:]
    y_test = y_train[train_test_sep_idx:]
    X_train = X_train[:train_test_sep_idx]
    y_train = y_train[:train_test_sep_idx]

    return X_train, y_train, X_test, y_test

def build_model(processor):
    logging.info('Build model...')

    action_input = Input(shape=(INPUT_LENGTH, INPUT_DIM), name='action_input')
    seq = [action_input] 
   
    # seq contains the outputs of every layer
    for i, dim in enumerate(INTER_DIM):
        if i == len(INTER_DIM) - 2:
            seq.append(LSTM(dim, return_sequences=True, dropout_W=0.2, dropout_U=0.2,
                consume_less=processor, name='second_to_last_output')(seq[-1]))
            second_to_last = seq[-1]
        else:
            seq.append(LSTM(dim, return_sequences=True, dropout_W=0.2, dropout_U=0.2,
                consume_less=processor)(seq[-1]))
    seq.append(TimeDistributed(Dense(OUTPUT_DIM))(seq[-1]))
    probs = Activation('softmax', name='prob_output')(seq[-1])

    # Output contains probabilities and input into layer before dense and softmax
    model = Model(input=action_input, output=[probs, second_to_last])

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  # metrics=['accuracy'],
                  loss_weights=[1., 0.])

    # plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)    
    return model

def train(model, X_train, y_train, X_test, y_test, start_weights_file=None):
    logging.info('Train...')
    save_weights = ModelCheckpoint('weights.{epoch:02d}.hdf5')
    logger = ProgbarLogger()
    printloss = PrintLoss(pad_ratio(y_test))

    if start_weights_file:
        model.load_weights(start_weights_file)

    dummy_train = np.zeros((y_train.shape[0], y_train.shape[1], INTER_DIM[-2]))
    dummy_test = np.zeros((y_test.shape[0], y_test.shape[1], INTER_DIM[-2]))

    model.fit(X_train, [y_train, dummy_train], batch_size=BATCH_SIZE,
              nb_epoch=NUM_EPOCHS,
              validation_data=(X_test, [y_test, dummy_test]),
              callbacks=[save_weights, logger, printloss])

    losses = model.evaluate(X_test, [y_test, dummy_test],
                            batch_size=BATCH_SIZE)
    loss = losses[0]
    logging.info('Scaled test loss: {0}'.format(loss*pad_ratio(y_test)))

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

    X_train, y_train, X_test, y_test = load_training_data()
    logging.info("Padding ratio (multiply loss by this): ")
    logging.info(pad_ratio(y_test))

    model = build_model(args.gpu)
    train(model, X_train, y_train, X_test, y_test, start_weights_file=args.file)
