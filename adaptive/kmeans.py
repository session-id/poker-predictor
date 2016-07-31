import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sys

def cluster(arr):
    kmeans = MiniBatchKMeans(n_clusters=5)

    return kmeans.fit_predict(arr)

def load(filename):
    with open(filename, 'r') as f:
        players = []
        stats = []
        for line in f.readlines():
            tokens = line.strip().split(',')
            players.append(tokens[0])
            stats.append(tokens[1:])

    return players, stats

if __name__ == '__main__':
    players, stats = load(sys.argv[1])
    clusters = cluster(stats)

    with open(sys.argv[2], 'w') as f:
        for player, cluster in zip(players, clusters):
            f.write(player + ',' + str(cluster) + '\n')