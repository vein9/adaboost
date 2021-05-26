import numpy as np
class LogisticRegresstion:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def sigmoid(S):
        """
        S: an numpy array
        return sigmoid function of each element of S
        """
        return 1 / (1 + np.exp(-S))

    def bias_trick(X):
        N = X.shape[0]
        return np.concatenate((X, np.ones((N, 1))), axis=1)

    def prob(self, w, X):
        """
        X: a 2d numpy array of shape (N, d). N datatpoint, each with size d
        w: a 1d numpy array of shape (d)
        """
        return self.sigmoid(X.dot(w))

    def loss(self, w, X, y, lam):
        """
        X, w as in prob 
        y: a 1d numpy array of shape (N). Each elem = 0 or 1 
        """
        z = self.prob(w, X)
        return -np.mean(y * np.log(z) + (1 - y) *
                        np.log(1 - z)) + 0.5 * lam / X.shape[0] * np.sum(w * w)

    def predict(self, w, X, threshold=0.5):
        """
        predict output of each row of X
        X: a numpy array of shape
        threshold: a threshold between 0 and 1 
        """
        res = np.zeros(X.shape[0])
        res[np.where(self.prob(w, X) > threshold)[0]] = 1
        return res

    def logistic_regression(self,
                            w_init,
                            X,
                            y,
                            lam=0.001,
                            lr=0.1,
                            nepoches=2000):
        # lam - regularization paramether, lr - learning rate, nepoches - number of epoches
        N, d = X.shape[0], X.shape[1]
        w = w_old = w_init
        # store history of loss in loss_hist
        loss_hist = [self.loss(w_init, X, y, lam)]
        ep = 0
        while ep < nepoches:
            ep += 1
            mix_ids = np.random.permutation(N)
            for i in mix_ids:
                xi = X[i]
                yi = y[i]
                zi = self.sigmoid(xi.dot(w))
                w = w - lr * ((zi - yi) * xi + lam * w)
            loss_hist.append(self.loss(w, X, y, lam))
            if np.linalg.norm(w - w_old) / d < 1e-6:
                break
            w_old = w
        return w, loss_hist