import argparse
import logging
import os
import sys
import json
import numpy as np

MAX_ACTIONS = 20
PLAYER_RANGE = (4, 7)

ACTION_MAP = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:3,
    5:3
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

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def get_input_vec(num_players, pos, action, street):
    input_vec = [0] * 17
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
        ind = street + 15
        input_vec[ind] = 1

    return input_vec

def get_output_vec(action):
    output_vec = [0] * 3
    if action is not None:
        if action == 2:
            action = 1
        elif action == 3:
            action = 2

        output_vec[action] = 1

    return output_vec

def parse_actions(actions, players_to_pos, num_players, street):
    street_ind = 0
    inputs = [get_input_vec(num_players, None, None, street_ind)]
    all_outputs = {player:[] for player in players_to_pos.keys()}
    for move in hand['actions']:
        if move == 'NEXT':
            if street == 'preflop' or street_ind == 1:
                break
            street_ind += 1
            continue

        pos = players_to_pos[move[0]]

        action = ACTION_MAP[move[1]]
        input_vec = get_input_vec(num_players, pos, action, street_ind)

        output_vec = get_output_vec(action)

        inputs.append(input_vec)
        for player, outputs in all_outputs.items():
            if player == move[0]:
                outputs.append(output_vec)
            else:
                outputs.append(get_output_vec(None))

    inputs = inputs[:-1]

    if len(inputs) < MAX_ACTIONS:
        for _ in range(MAX_ACTIONS - len(inputs)):
            inputs.append(get_input_vec(None, None, None, None))
            for outputs in all_outputs.values():
                outputs.append(get_output_vec(None))
    elif len(inputs) > MAX_ACTIONS:
        inputs = inputs[:MAX_ACTIONS]
        for player, outputs in all_outputs.items():
            all_outputs[player] = outputs[:MAX_ACTIONS]

    return inputs, all_outputs

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

def gen_training_data(hand, street):
    num_players = hand['num_players']

    # First filter by num players
    if num_players < PLAYER_RANGE[0] or num_players > PLAYER_RANGE[1]:
        return False

    if street == 'flop' and 'NEXT' not in hand['actions']:
        return False

    # Maps player to their position (UTG is 0, Mid is 1, etc.)
    players_to_pos = {}
    for i, move in enumerate(hand['actions'][:num_players]):
        if move == 'NEXT':
            return False
        players_to_pos[move[0]] = i

    # Parse Actions
    try:
        inputs, outputs = parse_actions(hand['actions'], players_to_pos, num_players, street)
    except KeyError:
        logging.info("Error with hand: " + str(hand))
        return False

    outputs = {player: outputs[pid]
               for player, pid in hand['players'].items()
               if pid in outputs}

    if args.street != 'preflop':
        if len(hand['board']) == 0:
            board = [[0] * 42] * MAX_ACTIONS
        else:
            board = [parse_board(hand['board'][:3])] * MAX_ACTIONS
        return inputs, board, outputs

    return inputs, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training data from JSONs.')
    parser.add_argument('input_dir', type=str, help="Input Directory")
    parser.add_argument('output_dir', type=str, help="Output Directory")
    parser.add_argument('street', type=str, help="Can be 'preflop', 'flop', or 'combined'")
    parser.add_argument('--tmp-dir', type=str, default='/tmp/', help="Specify temp folder.")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    players = set()
    for filename in os.listdir(input_dir):
        filename = os.path.join(input_dir, filename)
        logging.info("Parsing file: " + str(filename))

        with open(filename, 'r') as f:
            for hand in json.loads(f.read()):
                res = gen_training_data(hand, args.street)
                if not res:
                    continue

                inp = res[0] if args.street == 'preflop' else list(res[:2])
                outputs = res[-1]

                for player, out in outputs.items():
                    player = player.replace('/', '_')
                    mode = 'a' if player in players else 'w'
                    players.add(player)
                    g = open(os.path.join(args.tmp_dir, player + '.txt'), mode)
                    g.write('[{0},{1}]\n'.format(inp, out))
                    g.close()

    for player in players:
        input_file = os.path.join(args.tmp_dir, player + '.txt')
        out_file = os.path.join(args.output_dir, player)
        logging.info("Converting file to npz: " + str(input_file))
        with open(input_file, 'r') as f:
            inputs, outputs = [], []
            for line in f.readlines():
                inp, out = json.loads(line)
                inputs.append(inp)
                outputs.append(out)

            if args.street != 'preflop':
                inputs, boards = zip(*inputs)
                board_arr = np.asarray(boards)
            input_arr = np.asarray(inputs)
            output_arr = np.asarray(outputs)
            if input_arr.shape[-2:] != (20, 17):
                print input_arr.shape
                break

            if args.street != 'preflop':
                np.savez_compressed(out_file, input=input_arr, board=board_arr, output=output_arr)
            else:
                np.savez_compressed(out_file, input=input_arr, output=output_arr)
