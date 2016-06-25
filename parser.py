from os.path import *
import sys
import re
import json

MATCH_NUM = re.compile('\d+\.?\d*')

def extract(text, start, end):
    start_ind = text.index(start) + len(start)
    end_ind = text.index(end) if end else len(text)
    return text[start_ind:end_ind]

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

    id_counter = 0
    state = 'START'
    i = 0
    while i < len(lines):
        line = lines[i].replace(",", "")
        if not line:
            i += 1
            continue
        if line.startswith('*** SHOW DOWN ***') or 'not called' in line:
            return players, stacks, actions, board

        if state == 'START':
            if line.startswith('Seat'):
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

def parse(filename, output_dir):
    base = basename(filename)[:-4]
    if base.startswith('abs'):
        parser = parse_abs
    elif base.startswith('pty'):
        parser = parse_pty
    else:
        print "Cannot Parse"
        return

    with open(filename, 'r') as f:
        hands = f.read().replace("\r\n", "\n").split("\n\n")
        i = 0
        for hand in hands:
            res = parser(hand)
            if not res:
                continue
            players, stacks, actions, board = res
            text = json.dumps({
                'players': players,
                'stacks': stacks,
                'actions': actions,
                'board': board
            }, indent=4, separators=(',', ': '))

            output_base = base + '_' + str(i) + '.json'
            with open(join(output_dir, output_base), 'w') as g:
                g.write(text)

            i += 1

if __name__ == '__main__':
    output_dir = sys.argv[-1]
    for filename in sys.argv[1:-1]:
        parse(filename, output_dir)