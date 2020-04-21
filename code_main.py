import code_model
import code_word2vec
import pandas as pd
import keras
import pickle
import math
from sklearn import preprocessing

global_train_path = "train.csv"
global_test_path = "dev.csv"

# calculate RMSE
# take input of [[x],[x]...], [x,x,...]
def RMSE(Y_predict, Y_test):
    cnt = len(Y_predict)
    s = 0.0
    for i in range(0, cnt):
        s += math.pow(Y_predict[i][0] - Y_test[i], 2)
    return math.sqrt(s / cnt)


def run(is_first_time=True):
    # if first time run, assign last to True
    X_train, Y_train = code_model.model_data(global_train_path, True, is_first_time=False)
    X_test, Y_test = code_model.model_data(global_test_path, False, is_first_time=False)

    # normalize in format... in case
    maxlen = max(len(X_train[0]), len(X_test[0]))
    X_train = code_model.normalize(data=X_train, maxlen=maxlen)
    X_test = code_model.normalize(data=X_test, maxlen=maxlen)

    # Data normalize
    # FIXME: If we don't normalize the data, it will result in all prediction as a same number. 
    # 0.31229264 without scale recover
    # It's identically a case that the model didn't learn actual things... overfit or the cov is too small
    X_train = preprocessing.minmax_scale(X_train, feature_range=(0,maxlen-1), axis=1)
    X_test = preprocessing.minmax_scale(X_test, feature_range=(0,maxlen-1), axis=1)

    if is_first_time:
        model = code_model.train_biLSTM(X_train, Y_train, X_test, Y_test)
    else:
        # o = open("model32n.pkl", "rb")
        # model = pickle.load(o)
        # o.close()
        model = keras.models.load_model(code_model.global_model_path)

    predict_y = model.predict(X_test)
    predict_y = predict_y * 3.0

    Y_test = Y_test * 3.0

    rmse = RMSE(Y_predict=predict_y, Y_test=Y_test.tolist())
    print("RMSE = " + str(rmse))
    return 

run(False)