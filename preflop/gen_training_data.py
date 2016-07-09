import sys
import os
import json
import numpy as np

MAX_ACTIONS = 20
PLAYER_RANGE = (4, 7)

ACTION_MAP = {0:0,1:1,2:1,3:2,4:2,5:2}

def get_input_vec(num_players, pos, action):
    input_vec = [0] * 14
    if num_players is not None:
        ind = num_players - PLAYER_RANGE[0]
        input_vec[ind] = 1
    if pos is not None:
        ind = pos + 4
        input_vec[ind] = 1
    if action is not None:
        ind = action + 11
        input_vec[ind] = 1

    return input_vec

def get_output_vec(action):
    output_vec = [0] * 3
    if action is not None:
        output_vec[action] = 1

    return output_vec

def gen_training_data(hand):
    num_players = hand['num_players']

    # First filter by num players
    if PLAYER_RANGE[0] <= num_players <= PLAYER_RANGE[1]:
        # Maps player to their position (UTG is 0, Mid is 1, etc.)
        players_to_pos = {}
        for i, move in enumerate(hand['actions'][:num_players]):
            players_to_pos[move[0]] = i

        inputs = [get_input_vec(num_players, None, None)]
        outputs = []
        for move in hand['actions']:
            if move == 'NEXT':
                break

            pos = players_to_pos[move[0]]
            action = ACTION_MAP[move[1]]
            input_vec = get_input_vec(num_players, pos, action)
            output_vec = get_output_vec(action)

            inputs.append(input_vec)
            outputs.append(output_vec)

        inputs = inputs[:-1]

        if len(inputs) < MAX_ACTIONS:
            for _ in range(MAX_ACTIONS - len(inputs)):
                inputs.append(get_input_vec(None, None, None))
                outputs.append(get_output_vec(None))
        elif len(inputs) > MAX_ACTIONS:
            inputs = inputs[:MAX_ACTIONS]
            outputs = outputs[:MAX_ACTIONS]

        return inputs, outputs
    return False

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    inputs = []
    outputs = []
    BUCKET_SIZE = 100
    i = 0
    ind = 0
    for filename in os.listdir(input_dir):
        i += 1
        filename = os.path.join(input_dir, filename)
        print "Filename:", filename 

        f = open(filename, 'r')

        counter = 0
        for hand in json.loads(f.read()):
            try:
                res = gen_training_data(hand)
            except:
                continue
            if res:
                inp, out = res
                inputs.append(inp)
                outputs.append(out)
                counter += 1

        if i == BUCKET_SIZE:
            out_file = os.path.join(output_dir, 'training_'+str(ind)+'.npz')
            input_arr = np.asarray(inputs)
            output_arr = np.asarray(outputs)

            np.savez_compressed(out_file, input=input_arr, output=output_arr)
            i = 0
            inputs = []
            outputs = []
            ind += 1

        print "Num Hands: ", counter

        f.close()