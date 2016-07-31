import csv
import os
import numpy as np
import logging

DATA_DIR = "training_data"
MAX_FILES = 10000000000

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
                elif action == 1:
                    pass
            else:
                skip_next_input = False
            
            # Player's turn
            if np.max(o_vec) == 1:
                skip_next_input = True

def save_csv2(player_to_stats, filename):
    with open(filename, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"')
        for _, v in player_to_stats.iteritems():
            writer.writerow([_[:-4]] + list(v))

def save_csv(filename):
    with open(filename, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for _, v in player_to_stats.iteritems():
            writer.writerow([_] + list(v))

player_to_stats = {}

COLLECT_ACTION_PCT = False
COLLECT_VPIP_PFR = True

def compile_stats():
    logging.info('Loading data...')

    for i, filename in enumerate(os.listdir(DATA_DIR)):
        if i > MAX_FILES:
            break
        full_name = DATA_DIR + "/" + filename
        print full_name
        with open(full_name) as f:
            data = np.load(f)
            if len(data["input"].shape) == 3 and len(data["output"].shape) == 3:
                X = data["input"]
                y = data["output"]
                if COLLECT_ACTION_PCT:
                    probs = np.divide(np.sum(y, (0,1)).astype(np.float64),
                                      np.sum(y, (0,1,2)))
                    player_to_stats[filename] = probs
                elif COLLECT_VPIP_PFR:
                    # Assumes actions are (fold, check, call, raise)
                    actions = X[:,:,11:15]
                    if (np.sum(actions, (0,1,2)) < 100):
                        continue
                    probs = np.divide(np.sum(actions, (0,1)).astype(np.float64),
                                      np.sum(actions, (0,1,2)))
                    pfr = probs[3]
                    vpip = probs[3] + probs[2]
                    player_to_stats[filename] = np.array([vpip, pfr])

# compile_stats()
