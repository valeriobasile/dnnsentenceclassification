from conf import conf, input_dimensions
import spacy
import os
import logging as log



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
            labels.append(labels)
        except:
            log.warning("error reading line {0} of file {1}".format(len(sentences), training_file)),
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
        elif len(line.strip()) > 0:
            tokens.append(line.strip())
        else:
            sentences.append(tokens)
            tokens = []

    log.info("read {0} instances".format(len(sentences)))
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
        sentences_preprocessed = [preprocess(sentence, experiment, nlp) for sentence in sentences]
        log.info("writing preprocessed training set file")
        with open(training_preprocessed_file, "w") as fo:
            for sentence_preprocessed, label in zip(sentences_preprocessed, labels):
                fo.write("<label>{0}\n".format(label))
                for token in sentence_preprocessed:
                    fo.write("{0}\n".format(token))
                fo.write("\n")

    log.info("building vocabulary")
    vocabulary = build_vocabulary(sentences_preprocessed, experiment)

    log.info("vectorization")
    # TODO
    return vocabulary
