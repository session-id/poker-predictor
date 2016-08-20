import os
import numpy as np
import logging
import argparse
import json
import imp

import loaders

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poker Predictor.')
    parser.add_argument("--core", type=str, default="gpu1", help="Specify cpu, gpu core")
    parser.add_argument("model_folder", type=str, help="Model Folder")

    args = parser.parse_args()

    os.environ['THEANO_FLAGS'] = os.environ.get('THEANO_FLAGS', '') + ',device=' + args.core

from keras.callbacks import ModelCheckpoint, ProgbarLogger, Callback

np.random.seed(1337)  # for reproducibility
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

class Logger(Callback):
    def __init__(self, fp, num_epochs):
        self.fp = fp
        self.num_epochs = num_epochs
        self.total_train_loss = 0
        self.total_val_loss = 0
        self.total_train_acc = 0
        self.total_val_acc = 0
        self.count = 0

    def on_epoch_end(self, epoch, logs={}):
        logs['loss'] *= self.train_pad_ratio
        logs['val_loss'] *= self.test_pad_ratio
        logging.info('True Loss: {0}'.format(logs['val_loss']))
        self.fp.write('Epoch: {0}\n'
                      'Train Loss/Val Loss: {1}, {2}\n'
                      'Train Acc/Val Acc: {3}, {4}\n\n'
                      .format(epoch, logs['loss'], logs['val_loss'],
                              logs['acc'], logs['val_acc']))

        if epoch == self.num_epochs - 1:
            self.total_train_loss += logs['loss']
            self.total_val_loss += logs['val_loss']
            self.total_train_acc += logs['acc']
            self.total_val_acc += logs['val_acc']
            self.count += 1
            logging.info('Cumulative Average Loss: {0}'.format(self.total_val_loss / self.count))
            self.fp.write('Cum Train Loss/Val Loss: {0}, {1}\n'
                          'Cum Train Acc/Val Acc: {2}, {3}\n\n'
                          .format(self.total_train_loss / self.count,
                                  self.total_val_loss / self.count,
                                  self.total_train_acc / self.count,
                                  self.total_val_acc / self.count))

def pad_ratio(y):
    return float(y.shape[0] * y.shape[1]) / np.sum(y, (0,1,2))

def build_model(model_file):
    model_module = imp.load_source('model', model_file)
    return model_module.build_model()

def train(folder, config):
    train_cfg = config['train']
    training_folder = train_cfg['training_folder']
    params = ['group', 'randomize_hands', 'randomize_players', 'training_ratio', 
              'is_preflop', 'max_files', 'training_hands', 'min_testing_hands']
    params = {k: v for k, v in train_cfg.items() if k in params}

    logging.info('Load Training Data')
    training_data = loaders.load_training_data(training_folder, **params)

    X_train, X_test, y_train, y_test = training_data

    logging.info('Build Model')
    model = build_model(os.path.join(folder, 'model.py'))

    save_weights = ModelCheckpoint(os.path.join(folder, 'weights.{epoch:02d}.hdf5'))
    progbar = ProgbarLogger()
    fp = open(os.path.join(folder, 'log.txt'), 'w')
    logger = Logger(fp, train_cfg['epochs'])

    logging.info('Train...')
    if train_cfg['group']:
        if 'start_weights_file' in train_cfg:
            model.load_weights(train_cfg['start_weights_file'])

        logger.train_pad_ratio = pad_ratio(y_train)
        logger.test_pad_ratio = pad_ratio(y_test)

        model.fit(X_train, y_train, batch_size=train_cfg['batch_size'],
                  nb_epoch=train_cfg['epochs'], validation_data=(X_test, y_test),
                  callbacks=[save_weights, progbar, logger])
    else:
        if not train_cfg['is_preflop']:
            X_train = zip(*X_train)
            X_test = zip(*X_test)
        initial_weights = model.get_weights()
        for p_X_train, p_X_test, p_y_train, p_y_test in zip(X_train, X_test, y_train, y_test):
            p_X_train, p_X_test = list(p_X_train), list(p_X_test)

            if 'start_weights_file' in train_cfg:
                model.load_weights(train_cfg['start_weights_file'])
            else:
                model.set_weights(initial_weights)

            logger.train_pad_ratio = pad_ratio(p_y_train)
            logger.test_pad_ratio = pad_ratio(p_y_test)

            model.fit(p_X_train, p_y_train, batch_size=train_cfg['batch_size'],
                      nb_epoch=train_cfg['epochs'], validation_data=(p_X_test, p_y_test),
                      callbacks=[progbar, logger])

    fp.close()

if __name__ == '__main__':
    config = json.load(open(os.path.join(args.model_folder, 'config.json')))
    train(args.model_folder, config)