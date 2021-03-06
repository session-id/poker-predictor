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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poker Predictor.')
    parser.add_argument('--file', type=str, help='Start training weights.')
    parser.add_argument('--gpu', const='gpu', default='cpu', nargs="?", 
                        help='sum the integers (default: find the max)')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--max-training-files', type=int,
                        help='Maximum number of training files to use.')
    
    args = parser.parse_args()

    model.MAX_TRAINING_FILES = 3000
    cluster_to_data = model.load_training_data()

    logging.info("Finished loading training data. Finalizing training data.")
    
    for model_num in range(5):
        print "Testing with model from cluster " + str(model_num)
        weights_file = "weights-" + str(model_num) + ".09.hdf5"
        my_model = model.build_model(args.gpu)
        my_model.load_weights(weights_file)
        for cluster, data in cluster_to_data.iteritems():
            X, flops, y = data
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
            score, acc = my_model.evaluate([X, flops], y_test,
                                           batch_size=model.BATCH_SIZE)
            print "Cluster: " + str(cluster)
            print "loss, acc, scaled_loss", score, acc, score * model.pad_ratio(y)
