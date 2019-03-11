#!/usr/bin/env python

import sys
import logging as log
from conf import experiments, conf
from data import preprocess, read_file
import spacy
import os

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)

try:
    experiment = experiments[sys.argv[1]]
except:
    log.error("experiment \"{0}\" does not exist".format(sys.argv[1]))
    sys.exit(1)

nlp = spacy.load(experiment["language"], disable=["ner", "pos", "parser"])

data_dir = os.path.join(conf["data_dir"], experiment["data"])
data_file = os.path.join(data_dir, "{0}.tsv".format(sys.argv[2]))
sentences, labels = read_file(data_file)
sentences_preprocessed = [preprocess(sentence, experiment, nlp) for sentence in sentences]

log.info("writing preprocessed file")
for sentence_preprocessed, label in zip(sentences_preprocessed, labels):
    if label == sys.argv[3]:
        for token in sentence_preprocessed:
            sys.stdout.write("{0}\n".format(token))
