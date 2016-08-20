import os
import numpy as np
import argparse
import imp
import json
import logging

import loaders

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def build_model(model_file):
    model_module = imp.load_source('model', model_file)
    return model_module.build_model()

def predict(folder, config):
    predict_cfg = config['predict']
    predict_folder = predict_cfg['predict_folder']

    logging.info('Load Training Data')
    filenames = os.listdir(predict_folder)
    max_files = predict_cfg['max_files'] if 'max_files' in predict_cfg else None
    X, y = loaders.load_training_data(predict_folder, 
                                      is_preflop=predict_cfg['is_preflop'],
                                      load_all=True,
                                      max_files=max_files)

    logging.info('Build Model')
    model = build_model(os.path.join(folder, 'model.py'))
    model.load_weights(os.path.join(folder, predict_cfg['weights_file']))

    if not predict_cfg['is_preflop']:
        X = zip(*X)

    for p_X, p_y, filename in zip(X, y, filenames):
        p_X = list(p_X)
        output = model.predict(p_X)
        other_data = {
            predict_cfg['output_name']: output,
        }
        if predict_cfg['keep_target']:
            other_data['output'] = p_y
        if predict_cfg['keep_input']:
            other_data['input'] = p_X

        np.savez_compressed(os.path.join(predict_cfg['output_folder'], filename), **other_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poker Predictor.')
    parser.add_argument("--core", type=str, default="gpu1", help="Specify cpu, gpu core")
    parser.add_argument("model_folder", type=str, help="Model Folder")

    args = parser.parse_args()

    os.environ['THEANO_FLAGS'] = os.environ.get('THEANO_FLAGS', '') + ',device=' + args.core
    config = json.load(open(os.path.join(args.model_folder, 'config.json')))
    predict(args.model_folder, config)