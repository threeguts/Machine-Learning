import numpy as np

class LogisticRegressionSGA:
    def __init__(self, learning_rate=0.01, lambda_reg=0.01, n_epochs=10):
        self.learning_rate = learning_rate #η
        self.lambda_reg = lambda_reg  #λ 
        self.n_epochs = n_epochs    #Number of epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.random.randn(n) * 0.01  #Initialize weights to random small sizes

        for epoch in range(self.n_epochs):
            for i in range(m):
                xi = X[i]
                yi = y[i]
                gradient = (yi - self.sigmoid(np.dot(xi, self.weights))) * xi
                self.weights += self.learning_rate * (gradient + self.lambda_reg * self.weights)

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)
    