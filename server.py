import os
import json
import keras
import joblib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from cleaner import Cleaner
from flask_cors import CORS
from flask.json import jsonify
from config import TRAIN, MODELS, LOG_DIR
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
max_words = 5000
max_len = 200

sentiment = ['neutral', 'negative', 'positive']

app = Flask(__name__)
CORS(app, supports_credentials=True) #, resources={r"/infer": {"origins": "*"}})

@app.route("/")
def home():
    return render_template("index.html",result='infer')


@app.route("/train", methods=["GET"])
def train():

    if request.method == "GET":
        train_df = pd.read_csv(TRAIN)
        train_df = train_df[['selected_text', 'sentiment']]
        train_df.selected_text.fillna("No content", inplace=True)

        cleaner = Cleaner()
        sentences = train_df.selected_text.tolist()
        data = list()
        for sentence in sentences:
            sentence = clean_text(cleaner, sentence)
            data.append(sentence)

        data = np.array(data)
        y = []
        labels = np.array(train_df.sentiment)
        for i in range(len(labels)):
            if labels[i] == "neutral":
                y.append(0)
            elif labels[i] == "negative":
                y.append(1)
            elif labels[i] == "positive":
                y.append(2)

        labels = np.array(y)
        labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
        del y

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        joblib.dump(tokenizer, os.path.join(MODELS,"tokenizer.pkl"))

        sequences = tokenizer.texts_to_sequences(data)
        padded_sequences = pad_sequences(sequences, maxlen=max_len)
        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, random_state=0)

        train_model(X_train, X_test, y_train, y_test)
        return jsonify(
            result= "Training done"
        )
    else:
        return jsonify(
            result="Training cannot be done"
        )


@app.route("/infer", methods=["POST"])
def infer():
    if request.method == "POST":

        body = json.loads(request.data.decode("utf-8"))
        text = body.get("text", "")

        cleaner = Cleaner()
        text = clean_text(cleaner, text)

        tokenizer = joblib.load(os.path.join(MODELS,"tokenizer.pkl"))
        sequence = tokenizer.texts_to_sequences([text])
        test = pad_sequences(
            sequence, maxlen=max_len
        )

        _, model = create_BiLSTMRNN()
        model.load_weights(
            os.path.join(MODELS, 'BiLSTM.hdf5'))
        return jsonify(
            result= sentiment[
                np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]
        )
    else:
        return jsonify(
            result="POST API call is required"
        )


def train_model(X_train, X_test, y_train, y_test):
    chkpt, model = create_BiLSTMRNN()

    history = model.fit(
        X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[chkpt])

def create_BiLSTMRNN():
    model = Sequential()
    model.add(layers.Embedding(max_words, 40, input_length=max_len))
    model.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Implementing model checkpoins to save the best metric and do not lose it on training.
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS, "BiLSTM.hdf5"), monitor='val_accuracy', verbose=1, save_best_only=True,
                                  mode='auto', period=1, save_weights_only=False)
    return checkpoint, model

def clean_text(cleaner, sentence):
    sentence = cleaner.remove_url(sentence)
    sentence = cleaner.remove_emails(sentence)
    sentence = cleaner.remove_one_or_more_space_char(sentence)
    sentence = cleaner.remove_single_quotes(sentence)
    sentence = cleaner.remove_stopwords_punctuations(sentence)
    return sentence