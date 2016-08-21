from os.path import *
import sys
import re
import json
import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

MATCH_NUM = re.compile('\d+\.?\d*')

def extract(text, start, end):
    start_ind = text.index(start) + len(start)
    text = text[start_ind:]
    end_ind = text.index(end) if end else len(text)
    return text[:end_ind]

def parse_ong(text):
    def parse_action(line):
        player_ind = players[extract(line, '', ' ')]

        action = extract(line, ' ', '')
        if action.startswith('folds'):
            action = [0]
        elif action.startswith('checks'):
            action = [1]
        elif action.startswith('calls'):
            action = [2]
        elif action.startswith('bets'):
            amt = float(MATCH_NUM.findall(action)[0])
            action = [3, amt]
        elif action.startswith('raises'):
            amt = float(MATCH_NUM.findall(action)[1])
            action = [4, amt]
        else:
            return False

        return [player_ind] + action

    lines = text.split("\n")
    players = {}
    stacks = []
    actions = []
    board = []
    stakes = None

    id_counter = 0
    state = 'START'
    for line in lines:
        line = line.replace(",", "")
        if not line or line == '---':
            continue
        if line.startswith('Summary'):
            return stakes, players, stacks, actions, board

        if state == 'START':
            if line.startswith('Table'):
                stakes = float(extract(line.replace(",",""), '/$', ' Real'))
            elif line.startswith('Seat') and ':' in line:
                players[extract(line, ': ', ' (')] = id_counter

                stacks.append(float(extract(line, '($', ')')))
                id_counter += 1
            elif line.startswith('Dealing pocket cards'):
                state = 'PREFLOP'

        elif state == 'PREFLOP':
            if line.startswith('--- Dealing flop'):
                actions.append('NEXT')
                board.extend(extract(line, '[', ']').replace(",","").split(" "))
                for card in board:
                    assert len(card) in [2, 3]
                state = 'FLOP'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)

        elif state == 'FLOP':
            if line.startswith('--- Dealing turn'):
                actions.append('NEXT')
                board.append(extract(line, '[', ']'))
                state = 'TURN'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'TURN':
            if line.startswith('--- Dealing river'):
                actions.append('NEXT')
                board.append(extract(line, '[', ']'))
                state = 'RIVER'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'RIVER':
            res = parse_action(line)
            if res:
                actions.append(res)

    return False


def parse_pty(text):
    def parse_action(line):
        try:
            player_ind = players[extract(line, '', ' ')]
        except:
            return False
        action = extract(line, ' ', '')
        if action.startswith('folds'):
            action = [0]
        elif action.startswith('checks'):
            action = [1]
        elif action.startswith('calls'):
            action = [2]
        elif action.startswith('bets'):
            amt = float(MATCH_NUM.findall(action)[0])
            action = [3, amt]
        elif action.startswith('raises'):
            amt = float(MATCH_NUM.findall(action)[0])
            action = [4, amt]
        elif action.startswith('is all-In'):
            amt = float(MATCH_NUM.findall(action)[0])
            action = [5, amt]
        else:
            return False

        return [player_ind] + action

    lines = text.split("\n")
    players = {}
    stacks = []
    actions = []
    board = []
    stakes = None

    id_counter = 0
    state = 'START'
    for line in lines:
        line = line.replace(",", "")
        if not line:
            continue
        if 'show' in line:
            return stakes, players, stacks, actions, board
        if 'has left the table' in line:
            return False

        if state != 'START' and ': ' in line:
            continue

        if state == 'START':
            if line.startswith('$'):
                stakes = float(extract(line, '$', ' USD')) / 100
            elif line.startswith('Seat') and ':' in line:
                players[extract(line, ': ', ' (')] = id_counter

                stacks.append(float(extract(line, '( $', ' USD')))
                id_counter += 1
            elif line.startswith('** Dealing down cards **'):
                state = 'PREFLOP'

        elif state == 'PREFLOP':
            if line.startswith('** Dealing Flop **'):
                actions.append('NEXT')
                board.extend(extract(line, '[ ', ' ]').replace(",","").split(" "))
                for card in board:
                    assert len(card) in [2, 3]
                state = 'FLOP'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)

        elif state == 'FLOP':
            if line.startswith('** Dealing Turn **'):
                actions.append('NEXT')
                board.append(extract(line, '[ ', ' ]'))
                state = 'TURN'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'TURN':
            if line.startswith('** Dealing River **'):
                actions.append('NEXT')
                board.append(extract(line, '[ ', ' ]'))
                state = 'RIVER'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'RIVER':
            res = parse_action(line)
            if res:
                actions.append(res)

    return False

def parse_abs(text):
    def parse_action(line):
        try:
            player_ind = players[extract(line, '', ' ')]
        except:
            return False
        action = extract(line, ' - ', '')
        if action.startswith('Fold'):
            action = [0]
        elif action.startswith('Checks'):
            action = [1]
        elif action.startswith('Calls'):
            action = [2]
        elif action.startswith('Bets'):
            amt = float(MATCH_NUM.findall(action)[0])
            action = [3, amt]
        elif action.startswith('Raises'):
            amt = float(MATCH_NUM.findall(action)[1])
            action = [4, amt]
        elif action.startswith('All-In'):
            amt = float(MATCH_NUM.findall(action)[0])
            action = [5, amt]
        else:
            return False

        return [player_ind] + action

    lines = text.split("\n")
    players = {}
    stacks = []
    actions = []
    board = []
    stakes = None

    id_counter = 0
    state = 'START'
    for line in lines:
        line = line.replace(",", "")
        if not line:
            continue

        if line.startswith('*** SHOW DOWN ***') or 'not called' in line:
            return stakes, players, stacks, actions, board

        if state == 'START':
            if line.startswith('Stage'):
                if 'ante' in line:
                    stakes = float(extract(line, '$', ' $'))
                else:
                    stakes = float(extract(line, '$', ' - '))
            elif line.startswith('Seat'):
                try:
                    players[extract(line, '- ', ' (')] = id_counter
                except:
                    return False

                stacks.append(float(extract(line, '($', ' in chips')))
                id_counter += 1
            elif line.startswith('*** POCKET CARDS ***'):
                state = 'PREFLOP'

        elif state == 'PREFLOP':
            if line.startswith('*** FLOP ***'):
                actions.append('NEXT')
                board.extend(extract(line, '[', ']').replace(",","").split(" "))
                for card in board:
                    assert len(card) in [2, 3]
                state = 'FLOP'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'FLOP':
            if line.startswith('*** TURN ***'):
                actions.append('NEXT')
                board.append(line[line.rindex('[') + 1: line.rindex(']')])
                state = 'TURN'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'TURN':
            if line.startswith('*** RIVER ***'):
                actions.append('NEXT')
                board.append(line[line.rindex('[') + 1: line.rindex(']')])
                state = 'RIVER'
            else:
                res = parse_action(line)
                if res:
                    actions.append(res)
        elif state == 'RIVER':
            res = parse_action(line)
            if res:
                actions.append(res)

    return False

def parse(filename, output_dir, ind):
    base = basename(filename)[:-4]
    if base.startswith('abs'):
        parser = parse_abs
    elif base.startswith('pty'):
        parser = parse_pty
    elif base.startswith('ong'):
        parser = parse_ong
    else:
        logging.info("Cannot Parse")
        return False

    with open(filename, 'r') as f:
        hands = f.read().replace("\r\n", "\n").split("\n\n")
        output = []
        for i, hand in enumerate(hands):
            try:
                res = parser(hand)
            except:
                logging.info("Error with " + filename)
                continue

            if not res:
                continue
            stakes, players, stacks, actions, board = res

            actual_players = set()
            is_fold = True
            for action in actions:
                if action == "NEXT":
                    break
                is_fold = is_fold and action[1] == 0
                actual_players.add(action[0])

            if not is_fold:
                players = {k:v for k,v in players.items() if v in actual_players}

            output.append({
                'stakes': stakes, 
                'num_players': len(players), 
                'players': players,
                'stacks': stacks,
                'actions': actions,
                'board': board
            })

        text = json.dumps(output, indent=4, separators=(',', ': '))
        output_base = 'training_' + str(ind) + '.json'
        with open(join(output_dir, output_base), 'w') as g:
            g.write(text)

        return True

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    i = 0
    for filename in os.listdir(input_dir):
        logging.info("Parsing: " + filename)
        if parse(join(input_dir, filename), output_dir, i):
            i += 1