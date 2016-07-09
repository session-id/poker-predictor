import numpy as np
import model_1

class NaiveBayes:
    def __init__(self, num_classes, num_time_steps, num_features):
        self._tally_x = np.ones((num_classes, num_time_steps, num_features))
        self._tally_y = np.ones(num_classes)
        self._num_time_steps = num_time_steps
        self._num_features = num_features
        self._num_classes = num_classes

    # m is 2D matrix of histories with shape (n, num_features) where 1 <= n <= num_time_steps
    def add_history(self, class_, m):
        npad = ((self._num_time_steps - m.shape[0], 0), (0,0))
        m_padded = np.pad(m, pad_width=n_pad, mode='constant', constant_values=0)
        self._tally_x[class_] += m_padded
        self._tally_y[class_] += 1

    def end_training(self):
        self._log_p_x_bar_y = np.log(np.divide(self._tally_x, np.sum(self._tally_x, (2))))
        self._log_p_y = np.log(np.divide(self._tally_y, np.sum(self._tally_y)))

    def predict(self, m):
        log_probs = np.sum(np.multiply(self._log_p_x_bar_y, m), (1,2)) + self._log_p_y
        probs = np.exp(log_probs)
        probs = np.divide(probs, np.sum(probs))
        return probs


def train(y, nb):
    for hand in y:
        for i, action in enumerate(hand):
            # if hit padding
            if np.max(action) == 0:
                break
            class_ = list(action).index(1)
            nb.add_history(class_, hand[0:i, :])

if __name__ == '__main__':
    model_1.USE_ONE_TRAINING_FILE = True
    X_train, y_train, X_test, y_test = model_1.load_training_data()
    print("Training...")
    nb = NaiveBayes(X_train.shape[2], X_train.shape[1], X_train.shape[2])
    train(y_train, nb)