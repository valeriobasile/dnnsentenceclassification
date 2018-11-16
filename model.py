from keras.models import Sequential
from keras.layers import Activation, ActivityRegularization, Embedding, Dense, Bidirectional, Dropout, Flatten, LSTM, Conv1D, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import keras.backend as K

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def get_embedding_layer(experiment, word_index, embedding_matrix=None):
    if "embedding_file" in experiment:
        return Embedding(
            len(word_index)+1,
            experiment["embedding_dimension"],
            weights=[embedding_matrix],
            trainable=False,
            input_length=experiment["max_length"])
    else:
        return Embedding(
            len(word_index)+1,
            experiment["embedding_dimension"],
            input_shape=experiment["max_length"])

def create_model(experiment, X_train, y_train, embedding_matrix=None, word_index=None):
    model = Sequential()

    if experiment["model"] == "nn":
        model.add(Dense(200, input_shape=(X_train.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(ActivityRegularization(l1=0.0001, l2=0.00001))
        model.add(Dense(y_train.shape[1], activation='softmax'))
    elif experiment["model"] == "lstm":
        model.add(get_embedding_layer(experiment, word_index, embedding_matrix=embedding_matrix))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(ActivityRegularization(l1=0.001, l2=0.0001))
        model.add(Dense(y_train.shape[1], activation='softmax'))

    #model.summary()
    model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=[f1_score])
    return model

def train_model(model, X_train, y_train):
    callbacks = [EarlyStopping(monitor='val_loss', patience=1)]
    y_ints = [y.argmax() for y in y_train]
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)
    history = model.fit(X_train, y_train,
                            batch_size=100,
                            epochs=100,
                            verbose=1,
                            callbacks=callbacks,
                            validation_split=0.1,
                            class_weight=class_weights)
    return model
