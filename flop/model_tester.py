import model as m
import numpy as np
import sys

ACTION_TO_IND_MAP = {
    'check': 1,
    'call': 1,
    'bet': 2,
    'raise': 2,
    'fold': 0
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
    't':8,
    'j':9,
    'q':10,
    'k':11,
    'a':12
}

MAX_ACTIONS = m.INPUT_LENGTH
PLAYER_RANGE = (4, 7)

def new_settings():
    num_players = int(raw_input('Number of players? '))
    print "Options: reset (reset settings), nh (new hand), check, call, bet, raise, fold, flop Ah 5h 6s."

    return num_players

def new_hand(num_players):
    print "New Hand"
    return [False for _ in range(num_players)], 0, 0, [(None, None, None)], False

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

def predict(model, num_players, moves, turn, board, first_flop_action=False):
    moves_vec = [get_input_vec(num_players, pos, action, street) 
                 for pos, action, street in moves]
    moves_vec.extend([get_input_vec(None, None, None, None) 
                      for _ in range(MAX_ACTIONS - len(moves_vec))])
    moves_vec = np.array([moves_vec])

    board_vec = np.array([[parse_board(board)] * 20])

    output = model.predict([moves_vec, board_vec])[0]
    return output[turn] if first_flop_action else output[turn + 1]

def interact(model):
    num_players = new_settings()
    folded, turn, pos, moves, is_flop = new_hand(num_players)

    while True:
        inp = raw_input('--> ').lower().strip()
        if inp == 'reset':
            num_players = new_settings()
            folded, turn, pos, moves, is_flop = new_hand(num_players)
        elif inp == 'nh':
            folded, turn, pos, moves, is_flop = new_hand(num_players)
        elif inp.startswith('flop'):
            board = inp.split(" ")[1:]
            is_flop = True
            pos = -2 % num_players
            while True:
                pos = (pos + 1) % num_players
                if not folded[pos]:
                    break

            print predict(model, num_players, moves, turn, 
                          board, first_flop_action=True)
        else:
            if inp not in ACTION_TO_IND_MAP:
                continue
            action = ACTION_TO_IND_MAP[inp]
            if action == 0:
                folded[pos] = True

            if min(folded):
                folded, turn, pos, moves, is_flop = new_hand(num_players)
            else:
                moves.append((pos, action, is_flop))
                if is_flop:
                    print predict(model, num_players, moves, turn, board)
                turn += 1
                while True:
                    pos = (pos + 1) % num_players
                    if not folded[pos]:
                        break

if __name__ == '__main__':
    model = m.build_model('cpu')
    model.load_weights(sys.argv[1])
    interact(model)