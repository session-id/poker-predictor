import numpy as np
import model_1

def evaluate(inp, probs):
    num_players = list(inp[0,:4]).index(1) + 4
    most_probable_action = list(probs).index(max(list(probs)))

    num_correct = 0
    log_prob_sum = 0
    num_total = 0
    turn = 0
    folded = [False] * num_players
    for action in inp:
        if max(list(action[4:])) == 0:
            continue
        action = list(action[4:]).index(1)
        if action != 4 and action != 3:
            if action == most_probable_action:
                num_correct += 1
            log_prob_sum -= np.log(probs[action])
        else:
            num_correct += 1

        num_total += 1
        turn = (turn + 1) % num_players

    return (num_total, num_correct, log_prob_sum)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = model_1.load_training_data()
    probs = np.mean(y_train, (0,1))[0:3]
    probs = np.multiply(probs, 1.0 / np.sum(probs))
    print probs
    num_total = 0;
    num_correct = 0;
    log_prob_sum = 0;
    for hand in X_train:
        num_total_incr, num_correct_incr, log_prob_sum_incr = evaluate(hand, probs)
        num_total += num_total_incr
        num_correct += num_correct_incr
        log_prob_sum += log_prob_sum_incr
    print float(num_correct) / num_total
    print log_prob_sum / num_total