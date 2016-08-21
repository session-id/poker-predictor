from __future__ import print_function

import os
import numpy as np
import logging
import argparse

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ProgbarLogger, Callback

BATCH_SIZE = 64
NUM_EPOCHS = 5
INPUT_DIM = 17
INPUT_LENGTH = 20
OUTPUT_DIM = 3
INTER_DIM = (20, 10)
TRAINING_DATA_DIR = "../adaptive/training_data"
USE_ONE_TRAINING_FILE = False
TRAIN_DATA_RATIO = 0.75 # Amount of total data to use for training
CLUSTER_FILENAME = "../vpip_pfr_clusters.csv"

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
        for filename in files:
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

def load_training_data():
    logging.info('Loading data...')

    p_to_data = {}
    p_to_cluster = {}

    with open(CLUSTER_FILENAME) as f:
        for line in f:
            comma_idx = line.index(",")
            player_name = line[:comma_idx]
            cluster = int(line[comma_idx+1:])
            p_to_cluster[player_name] = cluster

    for i, filename in enumerate(p_to_cluster.keys()):
        if i > MAX_TRAINING_FILES:
            break
        full_name = TRAINING_DATA_DIR + "/" + filename
        if i % TRAINING_LOAD_PRINT_EVERY == 0:
            print full_name + "\r",
        with open(full_name) as f:
            data = np.load(f)
            if len(data["input"].shape) == 3 and len(data["output"].shape) == 3:
                X = data["input"]
                y = data["output"]

                p_to_data[filename] = (X, y)

    # This can be constructed earlier but it's very fast anyway
    cluster_to_p = defaultdict(lambda: [])
    for k, v in p_to_cluster.iteritems():
        cluster_to_p[v].append(k)

    cluster_to_data = {}
    for cluster, players in cluster_to_p.iteritems():
        Xs = []
        ys = []
        for player in players:
            if player in p_to_data:
                (X, y) = p_to_data[player]
                Xs.append(X)
                ys.append(y)
        if len(Xs) > 0:
            X_train = np.concatenate(tuple(Xs))
            y_train = np.concatenate(tuple(ys))
            cluster_to_data[cluster+CLUSTER_OFFSET] = (X_train, y_train)
        else:
            print "Empty Cluster:", cluster

    print "\n",

    return cluster_to_data

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


    cluster_to_data = load_training_data()

    logging.info("Finished loading training data. Finalizing training data.")

    models = []
    for cluster, data in cluster_to_data.iteritems():
        logging.info("Cluster " + str(cluster))
        if cluster in SKIP_CLUSTERS:
            continue;
        models.append(build_model(args.gpu))
        X, y = data

        train_test_sep_idx = int(X.shape[0] * TRAIN_DATA_RATIO)
        X_test = X[train_test_sep_idx:]
        y_test = y[train_test_sep_idx:]
        X_train = X[:train_test_sep_idx]
        y_train = y[:train_test_sep_idx]
        train(models[-1], X_train, y_train, X_test, y_test, cluster, start_weights_file=args.file)
