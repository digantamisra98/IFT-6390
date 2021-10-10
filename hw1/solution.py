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
        self.label_list = np.unique(train_labels)
        self.len_label = len(self.label_list)
        self.train_inputs=train_inputs
        self.train_labels=train_labels

    def pred_loop(self, predicted_class, counts):
        windows = []

        for i in range(0,len(self.test_data)):

            distance=(np.sum((np.abs(self.test_data[i] - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2)
            windows = np.array([j for j in range(len(distance)) if distance[j] < self.h])
            for window in windows:
                counts[i,int(self.train_labels[window]-1)]+=1
            if max(counts[i,:])==1:
                predicted_class[i]=draw_rand_label(self.test_data[i],self.label_list)
            else:
                predicted_class[i] = np.argmax(counts[i, :])+1
        return predicted_class

    def compute_predictions(self, test_data):
        
        self.test_data = test_data
        test_length = self.test_data.shape[0]
        predicted_class = np.zeros(test_length)
        counts=np.ones((test_length,self.len_label))

        return self.pred_loop(predicted_class,counts)


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.label_length = len(self.label_list)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def pred_loop(self, predicted_classes):

        for i in range(len(self.test_data)):
            distances = (np.sum((np.abs(self.test_data[i] - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2)
            dic = dict.fromkeys(self.label_list, 0)
            for l in range(len(distances)):
                sig=(1/(np.sqrt(2*np.pi)*self.sigma))*np.exp(-(distances[l]**2/(2*self.sigma**2)))*100

                dic[self.train_labels[l]]+=float(sig)

            predicted_classes[i]= max(dic, key=dic.get)
        return predicted_classes

    def compute_predictions(self, test_data):

        self.test_data = test_data
        length = test_data.shape[0]
        predicted_classes = np.zeros(length)

        return self.pred_loop(predicted_classes)
        


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
        
        hard_parzen_model= HardParzen(h)
        hard_parzen_model.train(self.x_train, self.y_train)
        y_pred = hard_parzen_model.compute_predictions(self.x_val)
        return ((self.y_val != y_pred).mean())
    
    def soft_parzen(self, sigma):
        
        soft_parzen_model = SoftRBFParzen(sigma)
        soft_parzen_model.train(self.x_train, self.y_train)
        y_pred = soft_parzen_model.compute_predictions(self.x_val)
        return ((self.y_val != y_pred).mean())


def get_test_errors(iris):
    train, val, test = split_dataset(iris)
    error_rate = ErrorRate(train[:,:-1], train[:,-1], val[:,:-1], val[:,-1])

    x_ticks = [0.01,0.1,0.2,0.3,0.4,0.5,1.0,3.0,10.0,20.0]
    y_hparzen = []
    y_sparzen = []

    for i in range (len(x_ticks)) :
        y_hparzen.append(error_rate.hard_parzen(x_ticks[i]))
        y_sparzen.append(error_rate.soft_parzen(x_ticks[i]))

    h_star = x_ticks[np.argmin(y_hparzen)]
    sigma_star = x_ticks[np.argmin(y_sparzen)]

    test_error = []
    error_rate_test_set = ErrorRate(train[:,:-1], train[:,-1], test[:,:-1], test[:,-1])
    test_error.append(error_rate_test_set.hard_parzen(h_star))
    test_error.append(error_rate_test_set.soft_parzen(sigma_star))

    return(np.array(test_error))

def random_projections(X, A):
    return np.dot(X,A)/np.sqrt(2)