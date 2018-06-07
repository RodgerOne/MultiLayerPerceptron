import numpy as np
import pickle as pkl


class SiecNeuronowa(object):
    def __init__(self, n_output, n_features, n_hidden=30, lambda_1=0.0, lambda_2=0.0, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.weight_1, self.weight_2 = self.initialize_weights()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        pass

    @staticmethod
    def save_data(data):
        PICKLE_FILE_PATH = 'neuron.pkl'
        with open(PICKLE_FILE_PATH, 'wb') as f:
            pkl.dump(data, f)
        pass

    @staticmethod
    def load_data():
        PICKLE_FILE_PATH = 'neuron.pkl'
        with open(PICKLE_FILE_PATH, 'rb') as f:
            data = pkl.load(f)
        return data

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def add_bias_unit(X, how ='columns'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1]+1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

    @staticmethod
    def regularization_lambda_1(lambda_param, weight_1, weight_2):
        return (lambda_param / 2.0) * (np.abs(weight_1[:, 1:]).sum() + np.abs(weight_2[:, 1:]).sum())

    @staticmethod
    def regularization_lambda_2(lambda_param, weight_1, weight_2):
        return (lambda_param / 2.0) * (np.sum(weight_1[:, 1:] ** 2) + np.sum(weight_2[:, 1:] ** 2))

    @staticmethod
    def onehot_convertion(y, categories):
        onehot = np.zeros((categories, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def print_model_params(self):
        print('n_hidden: ' + str(self.n_hidden) + ', epochs: ' + str(self.epochs) + ', minibatches: ' + str(self.minibatches))
        print('decrease_const: ' + str(self.decrease_const) + ', lambda_1: ' + str(self.lambda_1) + ', lambda_2: ' + str(self.lambda_2))
        print('eta: ' + str(self.eta) + ', alpha: ' + str(self.alpha) + ', shuffle: ' + str(self.shuffle))
        pass

    def initialize_weights(self):
        weight_1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        weight_1 = weight_1.reshape(self.n_hidden, self.n_features + 1)
        weight_2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        weight_2 = weight_2.reshape(self.n_output, self.n_hidden + 1)
        return weight_1, weight_2

    def feedforward(self, X, weight_1, weight_2):
        a_1 = self.add_bias_unit(X, how='column')
        z_2 = weight_1.dot(a_1.transpose())
        a_2 = self.sigmoid(z_2)
        a_2 = self.add_bias_unit(a_2, how='row')
        z_3 = weight_2.dot(a_2)
        a_3 = self.sigmoid(z_3)
        return a_1, z_2, a_2, z_3, a_3

    def cost_function(self, y_enc, output, weight_1, weight_2):
        term1 = -y_enc * np.log(output)
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        lambda_1_term = self.regularization_lambda_1(self.lambda_1, weight_1, weight_2)
        lambda_2_term = self.regularization_lambda_2(self.lambda_2, weight_1, weight_2)
        cost = cost + lambda_1_term + lambda_2_term
        return cost

    def gradient(self, a_1, a_2, a_3, z_2, y_enc, weight_1, weight_2):
        # propagacja wsteczna
        sigma_3 = a_3 - y_enc
        z_2 = self.add_bias_unit(z_2, how='row')
        sigma = SiecNeuronowa.sigmoid(z_2)
        sigma_2 = weight_2.T.dot(sigma_3) * (sigma * (1 - sigma))
        sigma_2 = sigma_2[1:, :]
        gradient_1 = sigma_2.dot(a_1)
        gradient_2 = sigma_3.dot(a_2.T)
        # regularyzacja
        gradient_1[:, 1:] += (weight_1[:, 1:] * (self.lambda_1 + self.lambda_2))
        gradient_2[:, 1:] += (weight_2[:, 1:] * (self.lambda_1 + self.lambda_2))
        return gradient_1, gradient_2

    def predict(self, X):
        return np.argmax(self.feedforward(X, self.weight_1, self.weight_2)[3], axis=0)

    def train_net(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = SiecNeuronowa.onehot_convertion(y, self.n_output)

        delta_w1_prev = np.zeros(self.weight_1.shape)
        delta_w2_prev = np.zeros(self.weight_2.shape)

        for i in range(self.epochs):
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                print('\rEpoch: %d/%d' % (i+1, self.epochs))

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                a_1, z_2, a_2, z_3, a_3 = self.feedforward(X_data[idx], self.weight_1, self.weight_2)
                cost = self.cost_function(y_enc=y_enc[:, idx], output=a_3, weight_1=self.weight_1, weight_2=self.weight_2)
                self.cost_.append(cost)
                grad1, grad2 = self.gradient(a_1=a_1, a_2=a_2, a_3=a_3, z_2=z_2, y_enc=y_enc[:, idx], weight_1=self.weight_1, weight_2=self.weight_2)
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.weight_1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.weight_2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        SiecNeuronowa.save_data(self) # do zapisu obiektu wyuczonej sieci
        return self
