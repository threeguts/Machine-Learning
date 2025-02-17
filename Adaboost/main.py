import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import Counter
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from AdaBoostClassifier import AdaBoost
from Stump import Stump
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  

def evaluate_model(model, X_train, y_train, X_test, y_test):  
    
    
    train_sizes = np.unique(np.logspace(2, np.log10(len(X_train)), 10).astype(int))

    precision_train, recall_train, f1_train = [], [], []
    precision_val, recall_val, f1_val = [], [], []

    for size in train_sizes:
        train_data = list(zip(X_train[:size], y_train[:size]))
        model.fit(train_data)
        y_train_pred = model.predict(X_train[:size])
        y_val_pred = model.predict(X_test)

        precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(
            y_train[:size], y_train_pred, average='macro', zero_division=0
        )
        precision_v, recall_v, f1_v, _ = precision_recall_fscore_support(
            y_test, y_val_pred, average='macro', zero_division=0
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

    #Recall plot
    axes[1].plot(train_sizes, recall_train, 'o-', label="Train Recall", color='blue')
    axes[1].plot(train_sizes, recall_val, 'x-', label="Dev Recall", color='orange')
    axes[1].set_title("Learning Curves: Recall")
    axes[1].set_ylabel("Recall")
    axes[1].legend()
    axes[1].grid(True)

    #F1-score plot
    axes[2].plot(train_sizes, f1_train, 'o-', label="Train F1-Score", color='blue')
    axes[2].plot(train_sizes, f1_val, 'x-', label="Dev F1-Score", color='orange')
    axes[2].set_title("Learning Curves: F1-Score")
    axes[2].set_xlabel("Training Size")
    axes[2].set_ylabel("F1-Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

#Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=None)

#Map numbers to words
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = {i + 3: word for word, i in word_index.items()}
index2word[0], index2word[1], index2word[2] = '[pad]', '[bos]', '[oov]'

#Convert numerical data to text (Optimized)
x_train = np.array([' '.join(map(index2word.get, text)) for text in x_train])
x_test = np.array([' '.join(map(index2word.get, text)) for text in x_test])

y_train = np.where(y_train == 0, -1, 1)
y_test_original = y_test.copy()
y_test = np.where(y_test == 0, -1, 1)


#Count word frequencies
word_counts = Counter(word for text in x_train for word in set(text.split()))

#Remove n most common and k least common words
print("removing n, k")
n, k = 400, 250
most_common = {word for word, _ in word_counts.most_common(n)}
least_common = {word for word, _ in word_counts.most_common()[:-k-1:-1]}
filtered_vocab = list(set(word_counts.keys()) - most_common - least_common)

print(f"Vocabulary size after filtering: {len(filtered_vocab)}")

#Convert texts to sparse binary vectors using CSR matrix
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
m = 200 
info_gain = mutual_info_classif(X_train_binary, y_train, discrete_features=True, n_jobs=-1)
top_m_indices = np.argsort(info_gain)[-m:]
final_vocab = [filtered_vocab[i] for i in top_m_indices]

print(f"Final vocabulary size: {len(final_vocab)}")

# 8. Create final binary vectors (Using only selected features)
X_train_final = X_train_binary[:, top_m_indices].toarray()  
X_test_final = X_test_binary[:, top_m_indices].toarray()    

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Evaluate the model
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("training adaboost")
train_data = list(zip(X_train_final, y_train))
adaboost = AdaBoost(L=Stump, M=100)
adaboost.fit(train_data)

# 10. Predict and evaluate
y_pred=adaboost.predict(X_test_final)
y_pred_converted=[1 if p == 1 else 0 for p in y_pred]
print(classification_report(y_test_original, y_pred_converted, zero_division=0))

evaluate_model(adaboost, X_train_final, y_train, X_test_final, y_test)
y_pred = adaboost.predict(X_test_final)
print(classification_report(y_test, y_pred))

