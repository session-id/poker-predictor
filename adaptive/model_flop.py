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

BATCH_SIZE = 500
NUM_EPOCHS = 10
INPUT_DIM = 16
FLOP_DIM = 42
OUTPUT_DIM = 3
INPUT_LENGTH = 20
INTER_DIM = (30, 10)
FLOP_INTER_DIM = (30, 20, 10)
TRAINING_DATA_DIR = "training_data"
TRAIN_DATA_RATIO = 0.75 # Amount of total data to use for training
CLUSTER_FILENAME = "m/clusters.csv"
SKIP_CLUSTERS = set([0, 1, 2])
TRAINING_LOAD_PRINT_EVERY = 100

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

    p_to_data = {}
    p_to_cluster = {}

    for i, filename in enumerate(os.listdir(TRAINING_DATA_DIR)):
        if i > MAX_TRAINING_FILES:
            break
        full_name = TRAINING_DATA_DIR + "/" + filename
        if i % TRAINING_LOAD_PRINT_EVERY:
            print full_name + "\r",
        with open(full_name) as f:
            data = np.load(f)
            if len(data["input"].shape) == 3 and len(data["output"].shape) == 3:
                X = data["input"]
                y = data["output"]
                flops = data["board"]

                p_to_data[filename[:-4]] = (X, flops, y)

    with open(CLUSTER_FILENAME) as f:
        for line in f:
            comma_idx = line.index(",")
            player_name = line[:comma_idx]
            cluster = int(line[comma_idx+1:])
            p_to_cluster[player_name] = cluster

    # This can be constructed earlier but it's very fast anyway
    cluster_to_p = defaultdict(lambda: [])
    for k, v in p_to_cluster.iteritems():
        cluster_to_p[v].append(k)

    cluster_to_data = {}
    for cluster, players in cluster_to_p.iteritems():
        Xs = []
        ys = []
        flops = []
        for player in players:
            if player in p_to_data:
                (X, flop, y) = p_to_data[player]
                Xs.append(X)
                ys.append(y)
                flops.append(flop)
        if len(Xs) > 0:
            X_train = np.concatenate(tuple(Xs))
            y_train = np.concatenate(tuple(ys))
            flops_train = np.concatenate(tuple(flops))

        cluster_to_data[cluster-1] = (X_train, flops_train, y_train)

    print "\n",

    return cluster_to_data

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

def train(model, X_train, y_train, X_test, y_test, cluster, start_weights_file=None):
    logging.info('Train...')
    save_weights = ModelCheckpoint('weights-'+str(cluster)+'.{epoch:02d}.hdf5')
    logger = ProgbarLogger()
    printloss = PrintLoss(pad_ratio(y_test))

    if start_weights_file:
        model.load_weights(start_weights_file)

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
              validation_data=(X_test, y_test), callbacks=[save_weights, logger, printloss])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=BATCH_SIZE)

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

    cluster_to_data = load_training_data()

    logging.info("Finished loading training data. Finalizing training data.")

    models = []
    for cluster, data in cluster_to_data.iteritems():
        logging.info("Cluster " + str(cluster))
        if cluster in SKIP_CLUSTERS:
            continue;
        models.append(build_model(args.gpu))
        X, flops, y = data
        new_flops = np.zeros((flops.shape[0], INPUT_LENGTH, flops.shape[1]))
        # Zero out flop before it comes out
        for i, X_hand in enumerate(X):
            for j, v in enumerate(X_hand):
                # First hand post-flop
                if v[15] == 1:
                    break
            new_flops[i] = np.concatenate((np.zeros((j, flops.shape[1])),\
                                           np.tile(np.expand_dims(flops[i], 0),\
                                                   (INPUT_LENGTH - j, 1))))

        flops = new_flops.astype(int)
        train_test_sep_idx = int(X.shape[0] * TRAIN_DATA_RATIO)
        X_test = X[train_test_sep_idx:]
        y_test = y[train_test_sep_idx:]
        X_train = X[:train_test_sep_idx]
        y_train = y[:train_test_sep_idx]
        flops_test = flops[train_test_sep_idx:]
        flops_train = flops[:train_test_sep_idx]
        train(models[-1], [X_train, flops_train], y_train, [X_test, flops_test], y_test, cluster, start_weights_file=args.file)
