# from __future__ import print_function

import os
import numpy as np
import logging
import argparse
import model

from collections import defaultdict

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM, merge, Input
from keras.callbacks import ModelCheckpoint, ProgbarLogger, Callback

np.random.seed(1337)  # for reproducibility

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def train(model_, X_train, y_train, X_test, y_test, start_weights_file=None):
    logging.info('Train...')
    save_weights = ModelCheckpoint('weights-generic.{epoch:02d}.hdf5')
    logger = ProgbarLogger()
    printloss = model.PrintLoss(model.pad_ratio(y_test))

    if start_weights_file:
        model_.load_weights(start_weights_file)

    model_.fit(X_train, y_train, batch_size=model.BATCH_SIZE, nb_epoch=model.NUM_EPOCHS,
              validation_data=(X_test, y_test), callbacks=[save_weights, logger, printloss])

    score, acc = model_.evaluate(X_test, y_test,
                                batch_size=model.BATCH_SIZE)

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
        model.BATCH_SIZE = args.batch_size

    if args.max_training_files:
        model.MAX_TRAINING_FILES = args.max_training_files

    cluster_to_data = model.load_training_data()

    logging.info("Finished loading training data. Finalizing training data.")

    Xs, flops, ys = zip(*cluster_to_data.values())
    X_con = np.concatenate(tuple(Xs))
    flops_con = np.concatenate(tuple(flops))
    y_con = np.concatenate(tuple(ys))

    model.NUM_EPOCHS = 5

    models = []
    if True:
        models.append(model.build_model(args.gpu))
        X, flops, y = X_con, flops_con, y_con
        new_flops = np.zeros((flops.shape[0], model.INPUT_LENGTH, flops.shape[1]))
        # Zero out flop before it comes out
        for i, X_hand in enumerate(X):
            for j, v in enumerate(X_hand):
                # First hand post-flop
                if v[15] == 1:
                    break
            new_flops[i] = np.concatenate((np.zeros((j, flops.shape[1])),\
                                           np.tile(np.expand_dims(flops[i], 0),\
                                                   (model.INPUT_LENGTH - j, 1))))

        flops = new_flops.astype(int)
        train_test_sep_idx = int(X.shape[0] * model.TRAIN_DATA_RATIO)
        X_test = X[train_test_sep_idx:]
        y_test = y[train_test_sep_idx:]
        X_train = X[:train_test_sep_idx]
        y_train = y[:train_test_sep_idx]
        flops_test = flops[train_test_sep_idx:]
        flops_train = flops[:train_test_sep_idx]
        train(models[-1], [X_train, flops_train], y_train, [X_test, flops_test], y_test, start_weights_file=args.file)
