import os
import sys
import json

import numpy as np

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

    for filename in input_files:
        print filename
        basename = os.path.basename(filename)[:-4]
        out_file = os.path.join(output_dir, basename + '.npz')
        with open(filename, 'r') as f:
            inputs, boards, outputs = [], [], []
            for line in f.readlines():
                line = "[" + line.strip() + "]"
                inp, board, out = json.loads(line)
                inputs.append(inp)
                boards.append(board)
                outputs.append(out)
            input_arr = np.asarray(inputs)
            board_arr = np.asarray(boards)
            output_arr = np.asarray(outputs)

            np.savez_compressed(out_file, input=input_arr, output=output_arr, board=board_arr)
