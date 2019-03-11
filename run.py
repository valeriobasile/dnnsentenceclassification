#!/usr/bin/env python

import sys
import logging as log
from conf import experiments
from data import load_data, load_embeddings
from model import create_model, train_model
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split, StratifiedKFold

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)

try:
    experiment = experiments[sys.argv[1]]
except:
    log.error("experiment \"{0}\" does not exist".format(sys.argv[1]))
    sys.exit(1)

X_train, X_test, y_train, y_test, word_index = load_data(experiment)

if "embedding_file" in experiment:
    embedding_matrix = load_embeddings(experiment, word_index)
    model = create_model(experiment, X_train, y_train, embedding_matrix=embedding_matrix, word_index=word_index)
else:
    model = create_model(experiment, X_train, y_train, word_index=word_index)

model = train_model(model, X_train, y_train)
evaluate_model(model, X_test, y_test)
#pred = model.predict_classes(X_test)
# for p in pred:
#     print (p)
