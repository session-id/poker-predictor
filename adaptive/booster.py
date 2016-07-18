import numpy as np
import math

def get_scores(models, inputs, desired_outputs):
    scores = []
    for model in models:
        guesses = model.predict(inputs)
        scores.append(np.sum(guesses * desired_outputs, 2).flatten())

    return np.array(scores)

def find_gradient(scores, weights):
    front_scores = scores[:-1]
    last_score = scores[-1]

    front_scores = front_scores - last_score

    denoms = last_score + weights.dot(front_scores)
    return front_scores.dot(1./denoms)

def find_weights(scores, steps=1000, lr=.02):
    N = len(scores)
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

def evaluate(models, all_inputs, all_desired_outputs):
    N = len(models)
    num_players = len(all_inputs)
    total_loss = 0
    for inputs, desired_outputs in zip(all_inputs, all_desired_outputs):
        scores = get_scores(models, inputs, desired_outputs)
        scores = np.transpose(np.array([scores[:,i] for i in range(scores.shape[1]) 
                              if sum(scores[:,i]) != 0]))

        weights = np.ones(N) * (1. / N)
        loss = -math.log(np.dot(weights, scores[:,0]))

        num_samples = scores.shape[1]
        for i in range(num_samples):
            weights = find_weights(scores[:,:i])
            loss += -math.log(np.dot(weights, scores[:,i]))

        loss = loss / num_samples
        total_loss += loss

    return total_loss / num_players

# EXAMPLE:
# import model
# model.USE_ONE_TRAINING_FILE = True
# X_train, flop_train, y_train, X_test, flop_test, y_test = model.load_training_data()
# m1 = model.build_model()
# m1.load_weights('weights.04.hdf5')
# m2 = model.build_model()
# m2.load_weights('weights.03.hdf5')
# models = [m1, m2]
# inputs = [X_test, flop_test]
# outputs = y_test
# print evaluate(models, [inputs], [outputs])
