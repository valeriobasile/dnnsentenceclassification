from conf import conf, input_dimensions
import spacy
import os
import logging as log
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

seed = 7
np.random.seed(seed)

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

def read_data_file(experiment, set):
    # read and preprocess the data
    data_dir = os.path.join(conf["data_dir"], experiment["data"])
    preprocessed_file = os.path.join(data_dir, "{0}_preprocessed.tsv".format(set))
    if os.path.isfile(preprocessed_file):
        log.info("{0} set file is already preprocessed, reading from {1}".format(set, preprocessed_file))
        sentences_preprocessed, labels = read_preprocessed_file(preprocessed_file)
    else:
        data_file = os.path.join(data_dir, "{0}.tsv".format(set))
        sentences, labels = read_file(data_file)
        if sentences == None:
            log.error("error reading file {0}, exiting".format(data_file))
        log.info("preprocessing the sentences")
        nlp = spacy.load(experiment["language"], disable=["ner", "pos", "parser"])

        try:
            sentences_preprocessed = [preprocess(sentence, experiment, nlp) for sentence in sentences]
        except:
            sentences_preprocessed = []
        log.info("writing preprocessed {0} set file".format(set))
        with open(preprocessed_file, "w") as fo:
            for sentence_preprocessed, label in zip(sentences_preprocessed, labels):
                fo.write("<label>{0}\n".format(label))
                for token in sentence_preprocessed:
                    fo.write("{0}\n".format(token))
                fo.write("\n")

    # transform the sentences into vectors
    log.info("vectorization")
    tokenizer = Tokenizer(filters='', lower=True, split=' ')
    tokenizer.fit_on_texts(sentences_preprocessed)
    word_index = tokenizer.word_index

    if experiment["wordrepresentation"] == 'tfidf':
        X_data = tokenizer.texts_to_matrix(sentences_preprocessed, mode='tfidf')
        X_data = pad_sequences(X_data, 8000, padding='post', truncating='post')
    elif experiment["wordrepresentation"] == 'embedding':
        X_data = tokenizer.texts_to_sequences(sentences_preprocessed)
        X_data = pad_sequences(X_data, experiment['max_length'])

    encoder = LabelBinarizer()
    #encoder = OneHotEncoder()
    encoder.fit(labels)
    y_data = encoder.transform(labels)
    y_data = to_categorical(encoder.transform(labels))

    return X_data, y_data, word_index


def load_data(experiment):
    data_dir = os.path.join(conf["data_dir"], experiment["data"])
    training_file = os.path.join(data_dir, "train.tsv")
    test_file = os.path.join(data_dir, "test.tsv")

    if not os.path.isfile(training_file):
        log.error("cannot find training file {0}, exiting".format(training_file))
        sys.exit(0)

    print (test_file, os.path.isfile(test_file))
    if os.path.isfile(test_file):
        log.error("reading training and test set")
        X_train, y_train, word_index = read_data_file(experiment, "train")
        X_test, y_test, _ = read_data_file(experiment, "test")
    else:
        log.error("reading training set and splitting")
        X_data, y_data, word_index = read_data_file(experiment, "train")
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=seed)

    return X_train, X_test, y_train, y_test, word_index

def load_embeddings(experiment, word_index):
    log.info("loading embeddings")
    embeddings_index = {}
    f = open('embeddings/{0}'.format(experiment["embedding_file"]))
    for line in f:
        values = line.strip().split()
        if len(values)<=2:
            continue
        word = " ".join(values[:-experiment['embedding_dimension']])
        coefs = np.asarray(values[-experiment['embedding_dimension']:], dtype='float32')
        embeddings_index[word.lower().strip()] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, experiment['embedding_dimension']))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
