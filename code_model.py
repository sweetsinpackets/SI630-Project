import code_word2vec
import code_preprocess
import sklearn
import keras
import pandas as pd
import numpy as np
import pickle

global_model_path = "model.h5"
global_history_path = "history.pkl"

# get data for training
# return X_train, Y_train, in format of dataframe
# FIXME: we scale the y to [0,1] and recover in predict
def model_data(path, is_train, is_first_time = True):
    col = code_word2vec.get_tovec(path, is_train, is_first_time)
    if is_train:
        p = "train_" + code_preprocess.global_tk_path
    else:
        p = "test_" + code_preprocess.global_tk_path
    data = pd.read_csv(p)
    # create a dataframe to match the type of two cases
    data["sentence_vec"] = col
    return data["sentence_vec"], data["meanGrade"] / 3.0


# normalize all data into a same format for the model
def normalize(data, maxlen):
    return keras.preprocessing.sequence.pad_sequences(sequences=data, maxlen=maxlen, dtype="float64", padding="post", truncating="post", value=0.0)


##****************************
## Since my computer having trouble running pytorch CUDA even I have...
## Hence I use colab+tensorflow to save my computer
## I use kersa to skip configuring layers
##****************************

def train_biLSTM(X_train, Y_train, X_test, Y_test):

    # build the bi-lstm model
    bi_lstm = keras.Sequential()
    # input_len is the size of sentence vector
    input_len = len(X_train[0])

    # embedding layer maps the input value into a index ranging by (0, input_len=300), 300 is the word2vec dimension
    # this is to reduce the training cost of LSTM to be affordable
    # meanwhile, it offers a int-scale discrete, hence we map the input value to range [0,300] also
    bi_lstm.add(keras.layers.Embedding(input_dim=input_len, output_dim=1))

    # the bi-lstm layer
    # the dropout/recurrent_dropout is tried to be optimal
    # bi_lstm.add(keras.layers.Bidirectional(keras.layers.LSTM    (units=input_len, dropout=0.15, recurrent_dropout=0.1)))
    bi_lstm.add(keras.layers.Bidirectional(keras.layers.LSTM    (units=input_len, activation="tanh", use_bias=True, dropout=0.15, recurrent_dropout=0.1)))

    # as usual... a fully-connected layer
    # this layer is to scale it to [0,1], and we will further scale...
    # Note that we scaled the Y_train to [0,1] also
    bi_lstm.add(keras.layers.Dense(units=1, activation="sigmoid"))

    # set how to study
    # we know that minimize mse is equal to minimize rmse
    bi_lstm.compile(loss='mse', optimizer='sgd', metrics=['accuracy', 'mse'])

    # print(bi_lstm.summary())

    # fit the model
    # since we don't have much data, small batch is better
    history_bi_lstm = bi_lstm.fit(X_train, Y_train, batch_size=32, epochs=5, validation_data=(X_test, Y_test))
    # history_bi_lstm = bi_lstm.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(X_test, Y_test))

    # TODO: delete it
    o = open("model32n.pkl", "wb")
    pickle.dump(bi_lstm, o)
    o.close()

    # save model
    bi_lstm.save(global_model_path)

    # save history pickle
    output = open(global_history_path, 'wb')
    pickle.dump(history_bi_lstm, output)
    output.close()

    # print the validation result of each epoch
    # note that it's not the actual mse, because we scaled the y
    print(history_bi_lstm.history)


    return bi_lstm


def load_biLSTM():
    return keras.models.load_model(global_model_path)


# the predict process...
def predict_biLSTM(X_test, model):
    return model.predict(X_test)
