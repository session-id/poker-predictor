import os
import numpy as np
import logging

DATA_DIR = ""
MAX_FILES = 100000000

USE_ONE_FILE = True
SINGLE_FILENAME = ""

# return [X_hand[0:4], flop[0:13], flop[13:26], \
#         flop[26:39], flop[39:42], X_hand[4:11]]

class PlayerStats:
    def __init__(self):
        self._num_hands = 0
        self._no_pressure_count = 0
        self._no_pressure_call = 0
        self._pressure_fold = 0
        self._pressure_call = 0
        self._pressure_count = 0

    def process2(self, X, y):
        self._num_hands += 1
        num_players_left = X[0,0:4].index(1)
        under_pressure = True
        skip_next_input = False
        for i_vec, o_vec in zip(X, y):
            # Might need to skip first vector
            
            if not skip_next_input:
                action = X[0,11:14].index(1)
                # Fold
                if action == 0:
                    num_players_left -= 1
                elif: action == 1:
                    pass
            else:
                skip_next_input = False
            
            # Player's turn
            if np.max(o_vec) == 1:
                skip_next_input = True

player_to_stats = {}

def compile_stats():
    logging.info('Loading data...')

    for i, filename in enumerate(os.listdir(DATA_DIR)):
        if i > MAX_FILES:
            break
        full_name = DATA_DIR + "/" + filename
        with open(full_name) as f:
            data = np.load(f)
            if len(data["input"].shape) == 3 and len(data["output"].shape) == 3::
                X = data["input"]
                y = data["output"]
                probs = np.divide(np.sum(y, (0)), np.sum(y, (0,1)))
                player_to_stats[filename] = probs


if __name__ == '__main__':
    compile_stats()
