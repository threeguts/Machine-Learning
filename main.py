import os

from sklearn.discriminant_analysis import StandardScaler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import Counter
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from logisticRegression import LogisticRegressionSGA 
from sklearn.linear_model import LogisticRegression
from bernoulli_nb import BernoulliNaiveBayes 
from sklearn.naive_bayes import BernoulliNB


def evaluate_model(model, X_train, y_train, X_test, y_test):  
    
    train_sizes = np.unique(np.logspace(3, np.log10(len(X_train)), 10).astype(int))

    precision_train, recall_train, f1_train = [], [], []
    precision_val, recall_val, f1_val = [], [], []

    for size in train_sizes:
        model.fit(X_train[:size], y_train[:size])  
        y_train_pred = model.predict(X_train[:size])
        y_val_pred = model.predict(X_test)

        precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(
            y_train[:size], y_train_pred, average='binary', pos_label=1, zero_division=0
        )
        precision_v, recall_v, f1_v, _ = precision_recall_fscore_support(
            y_test, y_val_pred, average='binary', pos_label=1, zero_division=0
        )

        precision_train.append(precision_t)
        recall_train.append(recall_t)
        f1_train.append(f1_t)
        precision_val.append(precision_v)
        recall_val.append(recall_v)
        f1_val.append(f1_v)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Precision plot
    axes[0].plot(train_sizes, precision_train, 'o-', label="Train Precision", color='blue')
    axes[0].plot(train_sizes, precision_val, 'x-', label="Dev Precision", color='orange')
    axes[0].set_title("Learning Curves: Precision")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].grid(True)

    # Recall plot
    axes[1].plot(train_sizes, recall_train, 'o-', label="Train Recall", color='blue')
    axes[1].plot(train_sizes, recall_val, 'x-', label="Dev Recall", color='orange')
    axes[1].set_title("Learning Curves: Recall")
    axes[1].set_ylabel("Recall")
    axes[1].legend()
    axes[1].grid(True)

    # F1-score plot
    axes[2].plot(train_sizes, f1_train, 'o-', label="Train F1-Score", color='blue')
    axes[2].plot(train_sizes, f1_val, 'x-', label="Dev F1-Score", color='orange')
    axes[2].set_title("Learning Curves: F1-Score")
    axes[2].set_xlabel("Training Size")
    axes[2].set_ylabel("F1-Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# 1. Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=None)

# 2. Map numbers to words
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = {i + 3: word for word, i in word_index.items()}
index2word[0], index2word[1], index2word[2] = '[pad]', '[bos]', '[oov]'

# 3. Convert numerical data to text (Optimized)
x_train = np.array([' '.join(map(index2word.get, text)) for text in x_train])
x_test = np.array([' '.join(map(index2word.get, text)) for text in x_test])

# 4. Count word frequencies
word_counts = Counter(word for text in x_train for word in set(text.split()))

# 5. Remove n most common and k least common words
n, k = 200, 300
most_common = {word for word, _ in word_counts.most_common(n)}
least_common = {word for word, _ in word_counts.most_common()[:-k-1:-1]}
filtered_vocab = list(set(word_counts.keys()) - most_common - least_common)

print(f"Vocabulary size after filtering: {len(filtered_vocab)}")

# 6. Convert texts to sparse binary vectors using CSR matrix
word_to_index = {word: i for i, word in enumerate(filtered_vocab)}
num_train, num_test, vocab_size = len(x_train), len(x_test), len(filtered_vocab)

train_rows, train_cols = [], []
for i, text in enumerate(x_train):
    for word in set(text.split()):
        if word in word_to_index:
            train_rows.append(i)
            train_cols.append(word_to_index[word])

X_train_binary = csr_matrix((np.ones(len(train_rows), dtype=np.int8), (train_rows, train_cols)), shape=(num_train, vocab_size))

test_rows, test_cols = [], []
for i, text in enumerate(x_test):
    for word in set(text.split()):
        if word in word_to_index:
            test_rows.append(i)
            test_cols.append(word_to_index[word])

X_test_binary = csr_matrix((np.ones(len(test_rows), dtype=np.int8), (test_rows, test_cols)), shape=(num_test, vocab_size))

# 7. Select m words with the highest information gain (Parallelized)
m = 7500 
info_gain = mutual_info_classif(X_train_binary, y_train, discrete_features=True, n_jobs=-1)
top_m_indices = np.argsort(info_gain)[-m:]
final_vocab = [filtered_vocab[i] for i in top_m_indices]

print(f"Final vocabulary size: {len(final_vocab)}")

# 8. Create final binary vectors (Using only selected features)
X_train_final = X_train_binary[:, top_m_indices].toarray()  
X_test_final = X_test_binary[:, top_m_indices].toarray()    

# 9. Train the model using your class 
model = BernoulliNaiveBayes(alpha=1.0)

# 10. Evaluate the model
evaluate_model(model, X_train_final, y_train, X_test_final, y_test)
y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))

model = LogisticRegressionSGA(learning_rate=0.01, lambda_reg=0.005, n_epochs=50) 

# 10. Evaluate the model
evaluate_model(model, X_train_final, y_train, X_test_final, y_test)
y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))

# Train the model with sklearn class
model = BernoulliNB(alpha=1.0)

# 10. Evaluate the model
evaluate_model(model, X_train_final, y_train, X_test_final, y_test)
y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))

# Train the model with sklearn class
# Κλίμακωση των δεδομένων για ομαλή σύγκλιση
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

model = LogisticRegression(
    penalty='l2',        # L2 regularization
    C=1/0.05,            # C = 1/λ
    solver='saga',       # Χρήση του saga (στοχαστική αναβάση κλίσης)
    max_iter=100,       # Αύξηση του αριθμού επαναλήψεων για καλύτερη σύγκλιση
    tol=1e-4,            # Ρυθμός ανοχής για το κριτήριο σύγκλισης
)

evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
