import model_1 as m
import numpy as np
import sys

ACTION_TO_IND_MAP = {
    'check': 1,
    'call': 1,
    'bet': 2,
    'raise': 2,
    'fold': 0
}

MAX_ACTIONS = m.INPUT_LENGTH
PLAYER_RANGE = (4, 7)

def new_settings():
    num_players = int(raw_input('Number of players? '))
    print "Options: reset (reset settings), nh (new hand), check, call, bet, raise, fold."

    return num_players

def new_hand(num_players):
    print "New Hand"
    return [False for _ in range(num_players)], 0, 0, [(None, None)]

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

def predict(model, num_players, moves, turn=None):
    arr = [get_input_vec(num_players, pos, action) for pos, action in moves]
    arr.extend([get_input_vec(num_players, None, None) for _ in range(MAX_ACTIONS - len(arr))])
    arr = np.array([arr])
    output = model.predict(arr)[0]
    return output[0] if turn is None else output[turn + 1]


def interact(model):
    num_players = new_settings()
    folded, turn, pos, moves = new_hand(num_players)
    print predict(model, num_players, moves)

    while True:
        inp = raw_input('--> ').lower().strip()
        if inp == 'reset':
            num_players = new_settings()
            folded, turn, pos, moves = new_hand(num_players)
            print predict(model, num_players, moves)
        elif inp == 'nh':
            folded, turn, pos, moves = new_hand(num_players)
            print predict(model, num_players, moves)
        else:
            if inp not in ACTION_TO_IND_MAP:
                continue
            action = ACTION_TO_IND_MAP[inp]
            if action == 0:
                folded[pos] = True

            if min(folded):
                folded, turn, pos, moves = new_hand(num_players)
                print predict(model, num_players, moves)
            else:
                moves.append((pos, action))
                print predict(model, num_players, moves, turn)
                turn += 1
                while True:
                    pos = (pos + 1) % num_players
                    if not folded[pos]:
                        break

if __name__ == '__main__':
    model = m.build_model('cpu')
    model.load_weights(sys.argv[1])
    interact(model)