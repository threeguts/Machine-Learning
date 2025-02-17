import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.class_probs = {}   # P(class)
        self.word_probs = {}    # P(word|class)
        self.alpha = alpha      # Laplace smoothing factor (gia na min exoume 0 probabilities)
        self.classes = None

    def fit(self, X, y):
        # upologismos twn P(class) kai P(word|class) gia kathe class :)
        total_docs = len(y)
        self.classes = np.unique(y)

       # P(class) = class_counts / total_docs
        class_counts = np.array([np.sum(y == c) for c in self.classes]) #posa documents exoume gia kathe class
        self.class_probs = class_counts / total_docs

        #  P(word|class) = (word_counts + alpha) / (class_counts + 2 * alpha)
        word_counts = np.array([X[y == c].sum(axis=0) for c in self.classes]) #poses fores emfanizetai to kathe word gia kathe class
        self.word_probs = (word_counts + self.alpha) / (class_counts[:, None] + 2 * self.alpha)

    def predict(self, X):
        #predict the class for each document!!!!! yay1!!
        log_class_probs = np.log(self.class_probs).reshape(1, -1)  # log(P(class))
        log_word_probs = np.log(self.word_probs)  # log(P(word|class))
        log_neg_word_probs = np.log(1 - self.word_probs)  # log(P(~word|class)) (vazoume logs gia apofigi praxewn me full mikrous arithmous)

        # upologismos tou log(P(word|class)) gia kathe class
        log_likelihoods = X @ log_word_probs.T + (1 - X) @ log_neg_word_probs.T  

        # prosthetoume to log(P(class)) kai to log(P(word|class)) gia kathe class
        log_posteriors = log_likelihoods + log_class_probs

        # epistrefoume thn klash me to megalutero log(P(class) * P(word|class))
        return self.classes[np.argmax(log_posteriors, axis=1)]
