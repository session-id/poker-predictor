import numpy as np
import model

if __name__ == '__main__':
    X_train, flops_train, y_train, X_test, flops_test, y_test = model.load_training_data()
    probs = np.mean(y_train, (0,1))
    probs = np.multiply(probs, 1.0 / np.sum(probs))
    print probs

    total_log_loss = -np.sum(np.multiply(y_test, np.log(probs)), (0,1,2))
    num_total = np.sum(y_test, (0,1,2))
    print y_test.shape, num_total
    print total_log_loss / num_total
