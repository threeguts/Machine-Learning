import tensorflow as tf
import numpy as np
import keras_preprocessing 
from collections import Counter
from gensim.models import KeyedVectors
from keras_preprocessing.sequence import pad_sequences  

def preprocess_texts(x_train, x_test, min_freq=5, max_vocab=20000):
   
    from collections import Counter

    word_counts = Counter()
    for sequence in x_train:
        word_counts.update(sequence)
    vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counts.most_common(max_vocab)) if count>=min_freq}

    return vocab

def load_word2vec_model(path_to_model):
    #φόρτωση Word2Vec model.
    return KeyedVectors.load_word2vec_format(path_to_model, binary=True)

def create_embedding_matrix(vocab, embedding_dim, word2vec_model):
    #εδώ γίνεται η υλοποίηση του Word2Vec model
    vocab_size = len(vocab)+1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, index in vocab.items():
        if word in word2vec_model:
            embedding_matrix[index]=word2vec_model[word]
        else:
            embedding_matrix[index]=np.random.uniform(-0.25, 0.25, embedding_dim)
    return embedding_matrix

def load_imdb_data(num_words=400, max_length=500):
    #φόρτωση IMDB dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

    return x_train, np.array(y_train), x_test, np.array(y_test)

def build_vocab(texts, min_freq=5, max_words=400):
    word_counts = Counter()
    
    for text in texts:
        word_counts.update(text.split())

    #φιλτράρουμε το λεξικό με βάση την συχνότητα και το όριο
    vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counts.most_common(max_words)) if count>=min_freq}

    return vocab