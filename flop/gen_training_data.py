import sys
import os
import json
import numpy as np

MAX_ACTIONS = 20
PLAYER_RANGE = (4, 7)

ACTION_MAP = {
    0:0,
    1:1,
    2:1,
    3:2,
    4:2,
    5:2
}

RANK_MAP = {
    '2':0,
    '3':1,
    '4':2,
    '5':3,
    '6':4,
    '7':5,
    '8':6,
    '9':7,
    '10':8,
    'T':8,
    'J':9,
    'Q':10,
    'K':11,
    'A':12
}

excess_count = 0

def get_input_vec(num_players, pos, action, street):
    input_vec = [0] * 16
    if num_players is not None:
        ind = num_players - PLAYER_RANGE[0]
        input_vec[ind] = 1
    if pos is not None:
        ind = pos + 4
        input_vec[ind] = 1
    if action is not None:
        ind = action + 11
        input_vec[ind] = 1
    if street is not None:
        ind = street + 14
        input_vec[ind] = 1

    return input_vec

def get_output_vec(action):
    output_vec = [0] * 3
    if action is not None:
        output_vec[action] = 1

    return output_vec

def parse_actions(actions, players_to_pos, num_players):
    global excess_count

    street = 0
    inputs = [get_input_vec(num_players, None, None, street)]
    outputs = []
    for move in hand['actions']:
        if move == 'NEXT':
            street += 1
            if street == 2:
                break
            continue

        pos = players_to_pos[move[0]]

        action = ACTION_MAP[move[1]]
        input_vec = get_input_vec(num_players, pos, action, street)

        if street == 0:
            output_vec = get_output_vec(None)
        else:
            output_vec = get_output_vec(action)

        inputs.append(input_vec)
        outputs.append(output_vec)

    inputs = inputs[:-1]

    if len(inputs) < MAX_ACTIONS:
        for _ in range(MAX_ACTIONS - len(inputs)):
            inputs.append(get_input_vec(None, None, None, None))
            outputs.append(get_output_vec(None))
    elif len(inputs) > MAX_ACTIONS:
        excess_count += 1
        inputs = inputs[:MAX_ACTIONS]
        outputs = outputs[:MAX_ACTIONS]

    return inputs, outputs

def parse_board(board):
    ranks = sorted([RANK_MAP[card[:-1]] for card in board])
    suits = sorted([card[-1] for card in board])

    rank_vec = []
    for rank in ranks:
        vec = [0] * 13
        vec[rank] = 1
        rank_vec.extend(vec)

    suit_vec = [0] * 3
    if suits[0] == suits[2]:
        suit_vec[0] = 1
    elif suits[0] == suits[1] or suits[1] == suits[2]:
        suit_vec[1] = 1
    else:
        suit_vec[2] = 1

    return rank_vec + suit_vec

def gen_training_data(hand):
    num_players = hand['num_players']

    # First filter by num players
    if num_players < PLAYER_RANGE[0] or num_players > PLAYER_RANGE[1]:
        return False

    if 'NEXT' not in hand['actions']:
        return False

    # Maps player to their position (UTG is 0, Mid is 1, etc.)
    players_to_pos = {}
    for i, move in enumerate(hand['actions'][:num_players]):
        players_to_pos[move[0]] = i

    try:
        inputs, outputs = parse_actions(hand['actions'], players_to_pos, num_players)
    except KeyError:
        print "Error with hand"
        print hand
        return False

    assert len(hand['board']) >= 3
    board = parse_board(hand['board'][:3])

    return inputs, board, outputs

if __name__ == '__main__':
    if len(sys.argv) == 2:
        input_files = []
        while True:
            try:
                line = raw_input()
                input_files.append(line)
            except EOFError:
                break
    else:
        input_files = sys.argv[1:-1]

    output_dir = sys.argv[-1]

    inputs = []
    outputs = []
    boards = []
    BUCKET_SIZE = 100
    i = 0
    ind = 0
    for filename in input_files:
        i += 1
        print "Filename:", filename 

        f = open(filename, 'r')

        counter = 0
        for hand in json.loads(f.read()):
            res = gen_training_data(hand)
            if res:
                inp, board, out = res
                inputs.append(inp)
                outputs.append(out)
                boards.append(board)
                counter += 1

        if i == BUCKET_SIZE:
            out_file = os.path.join(output_dir, 'training_'+str(ind)+'.npz')
            input_arr = np.asarray(inputs)
            output_arr = np.asarray(outputs)
            board_arr = np.asarray(boards)

            np.savez_compressed(out_file, input=input_arr, output=output_arr, board=board_arr)
            i = 0
            inputs = []
            outputs = []
            boards = []
            ind += 1

        print "Num Hands: ", counter
        print "Excess Count:", excess_count
        excess_count = 0 

        f.close()