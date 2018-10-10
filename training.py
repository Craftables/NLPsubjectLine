import pickle, random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def load_data():
    ldfile = open('dataset/title_data', mode='rb')
    x = pickle.load(ldfile)
    ldfile.close()

    ldfile = open('dataset/title_label', mode='rb')
    y = pickle.load(ldfile)
    ldfile.close()

    return x, y


def preprocess_data(x):
    count_vect = CountVectorizer(ngram_range=(1, 2))
    x_train_counts = count_vect.fit_transform(x)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    svfile = open('attributes/count_vect', mode='wb')
    pickle.dump(count_vect, svfile)
    svfile.close()

    svfile = open('attributes/tfidf_transformer', mode='wb')
    pickle.dump(tfidf_transformer, svfile)
    svfile.close()

    return x_train_tfidf


def start_training():
    x, y = load_data()

    x_train = list(x).copy()
    y_train = list(y).copy()

    x_train = preprocess_data(x_train)

    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    svfile = open('attributes/clf', mode='wb')
    pickle.dump(clf, svfile)
    svfile.close()
