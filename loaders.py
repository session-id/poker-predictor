import os
import numpy as np
import random

def load_training_data(folder, randomize_hands=False, randomize_players=False, 
                       group=False, training_ratio=0.75, is_preflop=True,
                       max_files=None, min_hands=150, load_all=False):
    fields = ['input', 'output']
    if not is_preflop:
        fields.insert(1, 'board')

    arrays = [[] for _ in range(len(fields))]

    files = os.listdir(folder)
    if max_files:
        files = files[:max_files]

    for filename in files:
        filename = os.path.join(folder, filename)
        with open(filename, 'r') as f:
            data = np.load(filename)
            for i, field in enumerate(fields):
                arrays[i].append(data[field])

    if randomize_hands:
        for j, player_arrays in enumerate(zip(*arrays)):
            perm = np.random.permutation(len(player_arrays[0]))
            for i, arr in enumerate(player_arrays):
                arrays[i][j] = arr[perm]

    if randomize_players:
        arrays = zip(*arrays)
        random.shuffle(arrays)
        arrays = zip(*arrays)

    if load_all:
        if group:
            for i, field in enumerate(arrays):
                arrays[i] = np.concatenate(field)

        if is_preflop:
            return arrays
        else:
            return [arrays[0], arrays[1]], arrays[2]

    new_arrays = []
    for i, field in enumerate(arrays):
        train = []
        test = []
        for player in field:
            if len(player) < min_hands:
                continue
            ind = int(len(player) * training_ratio)
            train.append(player[:ind])
            test.append(player[ind:])
        if group:
            train, test = [np.concatenate(x) for x in [train, test]]
        new_arrays.extend([train, test])

    if not is_preflop:
        rearrange = lambda x: [[x[0], x[2]], [x[1], x[3]], x[4], x[5]]
        new_arrays = rearrange(new_arrays)

    return new_arrays