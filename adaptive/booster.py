import numpy as np
import math
import os

from collections import defaultdict

def get_all_scores(models, all_inputs, all_flops, all_desired_outputs):
    all_scores = [[] for _ in range(len(all_inputs))]
    sizes = [len(inputs) for inputs in all_inputs]

    for model in models:
        all_guesses = model.predict([np.concatenate(tuple(all_inputs)), 
                                     np.concatenate(tuple(all_flops))])

        ind = 0
        for i, desired_outputs in enumerate(all_desired_outputs):
            guesses = all_guesses[ind:ind + sizes[i]]
            all_scores[i].append(np.sum(guesses * desired_outputs, 2).flatten())
            ind += sizes[i]

    return [np.array(scores) for scores in all_scores]

def find_gradient(scores, weights):
    front_scores = scores[:-1]
    last_score = scores[-1]

    front_scores = front_scores - last_score

    denoms = last_score + weights.dot(front_scores)
    return front_scores.dot(1./denoms)

def find_weights(scores, steps=1000, lr=.01, weights=None):
    N = len(scores)

    if weights is not None:
        weights = weights[:-1]
    else:
        weights = np.ones(N - 1) * (1. / N)

    for i in range(steps):
        new_weights = weights + lr * find_gradient(scores, weights)

        overflow = sum(new_weights) - 1
        if overflow > 0:
            new_weights -= overflow/(N - 1)

        for j, weight in enumerate(new_weights):
            if weight < 0:
                new_weights[j] = 0
            elif weight > 1:
                new_weights[j] = 1

        if sum(np.absolute(new_weights - weights)) < lr/100.:
            break

        weights = new_weights

    return np.append(weights, 1 - sum(weights))

def evaluate(models, all_inputs, all_flops, all_desired_outputs):
    N = len(models)
    num_players = len(all_inputs)
    total_loss = 0
    total_individual_losses = np.zeros(N)

    all_scores = get_all_scores(models, all_inputs, all_flops, all_desired_outputs)

    for scores, inputs, flops, desired_outputs in zip(all_scores, all_inputs, 
                                                      all_flops, all_desired_outputs):

        scores = np.array([scores[:,i] for i in range(scores.shape[1]) 
                          if sum(scores[:,i]) != 0])
        log_scores = np.log(scores)

        num_samples = scores.shape[0]

        choices = np.roll(np.argmax(np.cumsum(log_scores, axis=0), axis=1), 1)
        choices[0] = 0

        loss = 0
        for index, log_score in zip(choices, log_scores):
            loss -= log_score[index]
        loss /= float(num_samples)

        individual_losses = -np.mean(log_scores, (0,))

        # print "Loss:", loss
        # print "Individual Losses:", individual_losses

        total_loss += loss
        total_individual_losses += individual_losses

    return total_loss / num_players, total_individual_losses / num_players


INPUT_LENGTH = 20
TRAINING_DATA_DIR = 'training_data'
CLUSTER_FILENAME = 'm/clusters.csv'
TRAIN_DATA_RATIO = 0.25
MAX_TRAINING_FILES = 100000

def load_testing_data():
    p_to_data = {}
    p_to_cluster = {}

    for i, filename in enumerate(os.listdir(TRAINING_DATA_DIR)):
        if i > MAX_TRAINING_FILES:
            break
        full_name = TRAINING_DATA_DIR + "/" + filename

        with open(full_name) as f:
            data = np.load(f)
            if len(data["input"].shape) == 3 and len(data["output"].shape) == 3:
                X = data["input"]
                y = data["output"]
                flops = data["board"]

                p_to_data[filename[:-4]] = (X, flops, y)

    with open(CLUSTER_FILENAME) as f:
        for line in f:
            comma_idx = line.index(",")
            player_name = line[:comma_idx]
            cluster = int(line[comma_idx+1:])
            p_to_cluster[player_name] = cluster

    # This can be constructed earlier but it's very fast anyway
    cluster_to_p = defaultdict(lambda: [])
    for k, v in p_to_cluster.iteritems():
        cluster_to_p[v].append(k)

    cluster_to_data = {}
    for cluster, players in cluster_to_p.iteritems():
        Xs = []
        ys = []
        flops = []
        for player in players:
            if player in p_to_data:
                (X, flop, y) = p_to_data[player]
                Xs.append(X)
                ys.append(y)
                flops.append(flop)

        ind = int(len(Xs) * TRAIN_DATA_RATIO)
        Xs = Xs[ind:]
        flops = flops[ind:]
        ys = ys[ind:]

        cluster_to_data[cluster-1] = (Xs, flops, ys)

    return cluster_to_data

def main():
    import model

    models = [model.build_model('cpu') for _ in range(5)]
    for i, m in enumerate(models):
        m.load_weights('weights-' + str(i) + '.09.hdf5')

    cluster_to_data = load_testing_data()

    for cluster, data in cluster_to_data.iteritems():
        X, flops, y = data

        new_flops = [np.zeros((flop.shape[0], INPUT_LENGTH, flop.shape[1])) for flop in flops]

        # flops: (player, hand, board)
        # new_flops: (player, hand, actions, board)
        # X: (player, hand, actions, action)
        for i, player in enumerate(zip(flops, new_flops, X)):
            for j, (flop, new_flop, X_hand) in enumerate(zip(*player)):
                for k, v in enumerate(X_hand):
                    if v[15] == 1:   # determine if flop has been reached
                        break

                new_flops[i][j] = np.concatenate((np.zeros((k, flop.shape[0])),
                                                  np.tile(np.expand_dims(flop, 0),
                                                          (INPUT_LENGTH - k, 1))))

        flops = [x.astype(int) for x in new_flops]

        losses, individual_losses = evaluate(models, X, flops, y)
        print ("Total Cluster Loss for {n} players: {val}"
               .format(n=len(X), val=losses))
        print "Total Individual Losses:", individual_losses

if __name__ == '__main__':
    main()
