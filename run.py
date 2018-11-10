#!/usr/bin/env python

import sys
import logging as log
from conf import experiments
from data import load_data
from keras.models import Sequential
from keras.layers import Activation, ActivityRegularization, Embedding, Dense, Bidirectional, Dropout, Flatten, LSTM, Conv1D, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

seed = 7
np.random.seed(seed)

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)

def precision_recall(pred, gold, c):
    g = set([i for i, e in enumerate(gold) if e == c])
    p = set([i for i, e in enumerate(pred) if e == c])
    tp = len(g.intersection(p))
    if len(p) > 0:
        precision = float(tp)/float(len(p))
    else:
        precision = 0.0
    if len(g) > 0:
        recall = float(tp)/float(len(g))
    else:
        recall = 0.0

    fscore = (precision * recall * 2.0) / (precision + recall)
    return precision, recall, fscore

try:
    experiment = experiments[sys.argv[1]]
except:
    log.error("experiment \"{0}\" does not exist".format(sys.argv[1]))
    sys.exit(1)


X_data, y_data, word_index = load_data(experiment)
print (X_data.shape)
print (y_data.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=seed)

# pre trained embedding
log.info("loading embeddings")
embeddings_index = {}
f = open('wiki.it.vec')
for line in f:
    values = line.strip().split()
    if len(values)<=2:
        continue
    word = " ".join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word.lower().strip()] = coefs
f.close()
# f = open("wiki.it.bin", encoding="latin1")
# wv_from_bin = KeyedVectors.load_word2vec_format(f, binary=False)
# f.close()
#model = FastText.load_fasttext_format('wiki.it')
#print(model.wv['casa'])

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

if experiment["model"] == "nn":
    model = Sequential()
    model.add(Dense(100, input_shape=(X_data.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_data.shape[1], activation='softmax'))
elif experiment["model"] == "lstm":
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], trainable=False, input_length=140))
    #model.add(Embedding(len(word_index)+1, 256, input_shape=(140,)))
    #model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.5))
    #model.add(Bidirectional(LSTM(32)))
    #model.add(Dropout(0.5))
    model.add(ActivityRegularization(l1=0.001, l2=0.001))
    model.add(Dense(y_data.shape[1], activation='softmax'))

model.summary()
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

callbacks = [EarlyStopping(monitor='val_loss', patience=1),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(X_train, y_train,
                        batch_size=100,
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test,
                           batch_size=100, verbose=1)
pred = model.predict_classes(X_test)
gold = [np.argmax(i) for i in y_test]

p0, r0, f0 = precision_recall(pred, gold, 0)
p1, r1, f1 = precision_recall(pred, gold, 1)
macrof = (f0+f1)/2.0
print (p0, p1, r0, r1, f0, f1)
print ("macro F1-score: {0:.3f}".format(macrof))
print ("accuracy: {0:.3f}".format(score[1]))
