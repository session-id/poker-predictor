from os.path import *
import sys
import re
import json

MATCH_NUM = re.compile('\d+\.?\d*')

def extract(text, start, end):
    start_ind = text.index(start) + len(start)
    end_ind = text.index(end) if end else len(text)
    return text[start_ind:end_ind]

def parse_pty(text):
    pass

def parse_abs(text):
    def parse_action(line):
        player_ind = players[extract(line, '', ' - ')]
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
            print "Not an action:", action

        return [player_ind] + action

    lines = text.split("\n")
    players = {}
    stacks = []
    actions = []
    board = []
    stakes = None

    id_counter = 0
    state = 'START'
    i = 0
    while i < len(lines):
        line = lines[i].replace(",", "")
        if not line:
            i += 1
            continue
        if line.startswith('*** SHOW DOWN ***') or 'not called' in line:
            return stakes, players, stacks, actions, board

        if state == 'START':
            if line.startswith('Stage'):
                stakes = float(extract(line, '$', ' - '))
            elif line.startswith('Seat'):
                players[extract(line, '- ', ' (')] = id_counter

                stacks.append(float(extract(line, '($', ' in chips')))
                id_counter += 1
            elif line.startswith('*** POCKET CARDS ***'):
                state = 'PREFLOP'

        elif state == 'PREFLOP':
            if line.startswith('*** FLOP ***'):
                actions.append('NEXT')
                board.extend(extract(line, '[', ']').split(" "))
                state = 'FLOP'
            else:
                actions.append(parse_action(line))

        elif state == 'FLOP':
            if line.startswith('*** TURN ***'):
                actions.append('NEXT')
                board.append(line[-3:-1])
                state = 'TURN'
            else:
                actions.append(parse_action(line))
        elif state == 'TURN':
            if line.startswith('*** RIVER ***'):
                actions.append('NEXT')
                board.append(line[-3:-1])
                state = 'RIVER'
            else:
                actions.append(parse_action(line))
        elif state == 'RIVER':
            actions.append(parse_action(line))

        i += 1

    return False

def parse(filename, output_dir, ind):
    base = basename(filename)[:-4]
    if base.startswith('abs'):
        parser = parse_abs
    elif base.startswith('pty'):
        parser = parse_pty
    else:
        print "Cannot Parse"

    with open(filename, 'r') as f:
        hands = f.read().replace("\r\n", "\n").split("\n\n")
        output = []
        for hand in hands:
            try:
                res = parser(hand)
            except:
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
    for ind, filename in enumerate(input_files):
        print filename
        parse(filename, output_dir, ind)