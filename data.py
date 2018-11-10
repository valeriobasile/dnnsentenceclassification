from conf import conf, input_dimensions
import spacy
import os
import logging as log
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

def preprocess(sentence, experiment, nlp):
    doc = nlp(sentence)
    if experiment["preprocess"] == "tokens":
        return [token.text for token in doc]
    elif experiment["preprocess"] == "lemma":
        return [token.lemma_ for token in doc]

def build_vocabulary(sentences_preprocessed, experiment):
    index = 0
    vocabulary = {}
    for tokens in sentences_preprocessed:
        for token in tokens:
            if not token in vocabulary:
                vocabulary[token] = index
                index += 1
    return vocabulary

def read_file(data_file):
    try:
        f = open(data_file)
    except:
        log.error("cannot open file: {0}".format(data_file))
        return None, None

    sentences = []
    labels = []
    for line in f:
        try:
            sentence, label = line.strip().split("\t")
            sentences.append(sentence)
            labels.append(label)
        except:
            log.warning("error reading line {0} of file {1}".format(len(sentences), data_file)),
    log.info("read {0} instances".format(len(sentences)))
    return sentences, labels

def read_preprocessed_file(data_file):
    try:
        f = open(data_file)
    except:
        log.error("cannot open file: {0}".format(data_file))
        return None, None

    sentences = []
    labels = []
    tokens = []
    for line in f:
        if line.startswith("<label>"):
            label = line.strip().replace("<label>", "")
            labels.append(label)
        elif line != "\n":
            tokens.append(line.strip())
        else:
            sentences.append(tokens)
            tokens = []

    log.info("read {0} instances ({1} labels)".format(len(sentences), len(labels)))
    return sentences, labels

def load_data(experiment):
    data_dir = os.path.join(conf["data_dir"], experiment["data"])


    training_preprocessed_file = os.path.join(data_dir, "train_preprocessed.tsv")
    if os.path.isfile(training_preprocessed_file):
        log.info("training file is already preprocessed, reading from {0}".format(training_preprocessed_file))
        sentences_preprocessed, labels = read_preprocessed_file(training_preprocessed_file)
    else:
        training_file = os.path.join(data_dir, "train.tsv")
        sentences, labels = read_file(training_file)
        if sentences == None:
            log.error("error reading file {0}, exiting".format(training_file))
        log.info("preprocessing the sentences")
        nlp = spacy.load(experiment["language"], disable=["ner", "pos", "parser"])

        try:
            sentences_preprocessed = [preprocess(sentence, experiment, nlp) for sentence in sentences]
        except:
            sentences_preprocessed = []
        log.info("writing preprocessed training set file")
        with open(training_preprocessed_file, "w") as fo:
            for sentence_preprocessed, label in zip(sentences_preprocessed, labels):

                fo.write("<label>{0}\n".format(label))
                for token in sentence_preprocessed:
                    fo.write("{0}\n".format(token))
                fo.write("\n")

    log.info("vectorization")
    tokenizer = Tokenizer(filters='', lower=True, split=' ')
    tokenizer.fit_on_texts(sentences_preprocessed)
    #sequences = tokenizer.texts_to_sequences(sentences_preprocessed)
    word_index = tokenizer.word_index

    if experiment["wordrepresentation"] == 'tfidf':
        X_data = tokenizer.texts_to_matrix(sentences_preprocessed, mode='tfidf')
        X_data = pad_sequences(X_data, 8000, padding='post', truncating='post')
    elif experiment["wordrepresentation"] == 'embedding':
        X_data = tokenizer.texts_to_sequences(sentences_preprocessed)
        X_data = pad_sequences(X_data, 140)



    encoder = LabelBinarizer()
    encoder.fit(labels)
    y_data = to_categorical(encoder.transform(labels))

    return X_data, y_data, word_index
