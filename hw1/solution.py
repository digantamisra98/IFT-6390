import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

class Q1:

    def feature_means(self, iris):
        q = iris.mean(axis = 0).T[:-1]
        return q.T

    def covariance_matrix(self, iris):
        return np.cov(iris[:,:-1], rowvar = False)

    def feature_means_class_1(self, iris):
        feat = []
        for i in iris:
            if i.T[-1] == 1:
                feat.append(i[:][:-1])
        feat = np.array(feat)
        return feat.mean(axis = 0)

    def covariance_matrix_class_1(self, iris):
        feat = []
        for i in iris:
            if i.T[-1] == 1:
                feat.append(i[:][:-1])
        feat = np.array(feat)
        return np.cov(feat, rowvar = False)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def compute_predictions(self, test_data):
        pass


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def compute_predictions(self, test_data):
        pass


def split_dataset(iris):
    train, val, test = [],[],[]
    for i in range(iris.shape[0]):
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            train.append(iris[i])
        elif i % 5 == 3:
            val.append(iris[i])
        else:
            test.append(iris[i])
    return (np.array(train), np.array(val), np.array(test))


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


def random_projections(X, A):
    return np.dot(X,A)/np.sqrt(2)
