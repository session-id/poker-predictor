# from __future__ import print_function

import os
import numpy as np
import logging
import argparse

from collections import defaultdict

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM, merge, Input
from keras.callbacks import ModelCheckpoint, ProgbarLogger, Callback

BATCH_SIZE = 5
NUM_EPOCHS = 10
INPUT_DIM = 5
OUTPUT_DIM = 3
INPUT_LENGTH = 20
INTER_DIM = (3, )
TRAINING_DATA_DIR = "training_data"
TRAIN_DATA_RATIO = 0.75 # Amount of total data to use for training
TRAINING_LOAD_PRINT_EVERY = 100
CLUSTER_OFFSET = 0 # Amount to add to written cluster index to get actual index
MIN_HANDS = 100

SINGLE_TRAINING_FILENAME =  "training_data/training_2.npz"
USE_ONE_TRAINING_FILE = False
MAX_TRAINING_FILES = 10000000

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
    Xs = []
    ys = []
    files = os.listdir(TRAINING_DATA_DIR)
    np.random.shuffle(files)
    for i, filename in enumerate(files):
        if i == 1:
            continue
        if i >= MAX_TRAINING_FILES:
            break
        full_name = TRAINING_DATA_DIR + "/" + filename
        with open(full_name) as f:
            data = np.load(f)
            Xs.append(data["input"])
            ys.append(data["output"])

    return Xs, ys

def build_model(processor):
    logging.info('Build model...')

    action_input = seq = Input(shape=(INPUT_LENGTH, INPUT_DIM))
        
    for dim in INTER_DIM:
        seq = LSTM(dim, return_sequences=True, dropout_W=0.2, dropout_U=0.2,
                   consume_less=processor)(seq)
    seq = TimeDistributed(Dense(OUTPUT_DIM))(seq)
    probs = Activation('softmax')(seq)

    model = Model(input=action_input, output=probs)

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)    
    return model

def train(model, X, y):
    logger = ProgbarLogger()

    if len(X) < MIN_HANDS:
        return False

    X_train = X[:MIN_HANDS]
    y_train = y[:MIN_HANDS]
    X_test = X[MIN_HANDS:]
    y_test = y[MIN_HANDS:]

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
              callbacks=[logger])

    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    logging.info('Loss: ' + str(loss))
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poker Predictor.')
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

    Xs, ys = load_training_data()

    model = build_model(args.gpu)

    logging.info('Train...')
    total_loss = 0
    count = 0
    for X, y in zip(Xs, ys):
        loss = train(model, X, y)
        if loss:
            total_loss += loss
            count += 1

    logging.info('Total Loss: ' + str(total_loss / count))
