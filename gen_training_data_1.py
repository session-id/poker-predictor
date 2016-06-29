import sys
import os
import json
import numpy as np

MAX_ACTIONS = 20
PLAYER_RANGE = (4, 7)

def init_vec(num_players):
    input_vec = [0]*8
    input_vec[num_players - PLAYER_RANGE[0]] = 1
    return input_vec

def gen_training_data(hand):
    num_players = hand['num_players']
    if PLAYER_RANGE[0] <= num_players <= PLAYER_RANGE[1]:
        players = []
        for action in hand['actions'][:num_players]:
            players.append(action[0])

        i = 0
        j = 0
        count = 0

        inputs = [init_vec(num_players)]
        outputs = []
        while i < len(hand['actions']):
            action = hand['actions'][i]
            if action == 'NEXT':
                break

            input_vec = init_vec(num_players)
            output_vec = [0] * 4

            if action[0] == players[j]:
                if action[1] == 0:
                    input_vec[4] = 1
                    output_vec[0] = 1
                elif action[1] in [1, 2]:
                    input_vec[5] = 1
                    output_vec[1] = 1
                else:
                    input_vec[6] = 1
                    output_vec[2] = 1
                i += 1
            else:
                input_vec[7] = 1
                output_vec[3] = 1
                count += 1
                if count == 10:
                    return False

            inputs.append(input_vec)
            outputs.append(output_vec)

            j = (j + 1) % num_players

        inputs = inputs[:-1]
        if len(inputs) < MAX_ACTIONS:
            for _ in range(MAX_ACTIONS - len(inputs)):
                inputs.append([0]*8)
                outputs.append([0, 0, 0, 1])
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
            res = gen_training_data(hand)
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