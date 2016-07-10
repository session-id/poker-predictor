import numpy as np
import model

class SequenceNaiveBayes:
    def __init__(self, num_classes, num_time_steps, extra_dims):
        self._tally_x = np.ones((num_classes, num_time_steps, num_classes))
        self._tally_y = np.ones(num_classes)
        self._num_time_steps = num_time_steps
        self._num_classes = num_classes

        # Extra dims are nontimeseries data
        self._tally_x2 = []
        for extra_dim in extra_dims:
            self._tally_x2.append(np.ones((num_classes, extra_dim)))
        self._extra_dims = extra_dims

    # m is 2D matrix of histories with shape (n, num_classes) where 1 <= n <= num_time_steps
    def add_history(self, class_, m, extras):
        npad = ((self._num_time_steps - m.shape[0], 0), (0,0))
        m_padded = np.pad(m, pad_width=npad, mode='constant', constant_values=0)
        self._tally_x[class_] += m_padded
        for i, extra in enumerate(extras):
            self._tally_x2[i][class_] += extra
        self._tally_y[class_] += 1

    def end_training(self):
        summed_tally = np.tile(np.expand_dims(np.sum(self._tally_x, (2)), axis=2), \
                               (1,1,self._num_classes))
        self._log_p_x_bar_y = np.log(np.divide(self._tally_x, summed_tally))
        self._log_p_y = np.log(np.divide(self._tally_y, np.sum(self._tally_y)))

        self._log_p_x2_bar_y = []
        for tally, extra_dim in zip(self._tally_x2, self._extra_dims):
            summed_tally = np.tile(np.expand_dims(np.sum(tally, (1)), axis=1), \
                                   (1,extra_dim))
            self._log_p_x2_bar_y.append(np.log(np.divide(tally, summed_tally)))

    def predict(self, m, extras):
        npad = ((self._num_time_steps - m.shape[0], 0), (0,0))
        m_padded = np.pad(m, pad_width=npad, mode='constant', constant_values=0)
        log_probs = np.sum(np.multiply(self._log_p_x_bar_y, m_padded), (1,2)) \
                    + self._log_p_y
        for log_p_x2_bar_y, extra in zip(self._log_p_x2_bar_y, extras):
            log_probs += np.sum(np.multiply(log_p_x2_bar_y, extra), (1))
        probs = np.exp(log_probs)
        probs = np.divide(probs, np.sum(probs))
        return probs

def train(y, X, nb):
    for n, (hand, meta) in enumerate(zip(y, X)):
        print "\r" + str(n+1) + "/" + str(y.shape[0]),
        extra = meta[0, 0:11]
        for i, action in enumerate(hand):
            # if hit padding
            if np.max(action) == 0:
                continue
            class_ = list(action).index(1)
            nb.add_history(class_, hand[0:i, :], [extra])

def test(y, X, nb):
    num_total = 0
    total_log_loss = 0
    for n, (hand, meta) in enumerate(zip(y, X)):
        print "\r" + str(n+1) + "/" + str(y.shape[0]),
        extra = meta[0, 0:11]
        for i, action in enumerate(hand):
            if np.max(action) == 0:
                continue
            probs = nb.predict(hand[0:i, :], [extra])
            total_log_loss -= np.log(np.sum(np.multiply(probs, action)))
            num_total += 1
    print("Average log loss:")
    print(total_log_loss / num_total)

if __name__ == '__main__':
    # model.USE_ONE_TRAINING_FILE = True
    X_train, flop_train, y_train, X_test, flop_test, y_test = model.load_training_data()
    print("Training...")
    nb = SequenceNaiveBayes(y_train.shape[2], y_train.shape[1], [11])
    train(y_train, X_train, nb)
    nb.end_training()
    print("\n")
    print("Testing...")
    test(y_test, X_test, nb)
