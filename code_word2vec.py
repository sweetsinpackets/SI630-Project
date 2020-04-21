import code_preprocess
import pandas as pd
import numpy as np
import re
import nltk
import string
import pickle
# if encounter "unable to import 'smart_open.gcs', disabling that module", downgrade "pip install smart_open==1.10.0"
# but it seems doesn't matter....
from gensim.models import KeyedVectors

# we seperate storage of data and vector because saving and reading np.array(np.array(np.array)) will cause it turns to string...
global_w2v_npy = "wordvec.pkl"
global_s2v_npy = "sen2vec.pkl"

# it will create a column, return the word 2 vector as a list
def data_word2vec(path, is_train, is_first_time=True):
    # read preprocessed data
    data = code_preprocess.get_preprocessed_data(path, is_train, is_first_time)
    
    # load model
    # you need to have the model file, which is a pre-trained model
    w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # match word to the vector
    vector = []
    for (_, row) in data.iterrows():
        temp = []
        for i in row["tokenized"]:
            if i in w2v_model:
                temp.append(w2v_model[i])
        # vector for a row, in order not to modify dataframe everytime, enhance speed
        vector.append(temp)

    if is_train:
        wp = "train_" + global_w2v_npy
    else:
        wp = "test_" + global_w2v_npy

    output = open(wp, 'wb')
    pickle.dump(vector, output)
    output.close()
    return vector



# we represent the vector of sentence as a vector
# current method is to use average of all word vectors
# note that there may have dimension mismatch, so we force a dimension
# only return the list of sentence vector
def data_sentence2vec(path, is_train, is_first_time = True):
    if is_first_time:
        vector = data_word2vec(path, is_train, True)
    else:
        if is_train:
            wp = "train_" + global_w2v_npy
        else:
            wp = "test_" + global_w2v_npy        
        pkl_file = open(wp, 'rb')
        vector = pickle.load(pkl_file)
        pkl_file.close()
    
    col_array = []  # as the column

    for row in vector:
        #data["wordvec"][0][0] is the first wordvector of first row in dataframe
        l = np.zeros(len(vector[0][0]))
        cnt = 0
        # add all word vectors
        for i in row:
            l = np.add(l, i)    
            cnt += 1
        # average
        for i in range(0, len(vector[0][0])):
            l[i] = l[i] / cnt
        col_array.append(l)

    if is_train:
        sp = "train_" + global_s2v_npy
    else:
        sp = "test_" + global_s2v_npy

    output = open(sp, 'wb')
    pickle.dump(col_array, output)
    output.close()
    return col_array


# a top level function
def get_tovec(path, is_train, is_first_time = True):
    if is_first_time:
        return data_sentence2vec(path, is_train, True)
    else:
        if is_train:
            sp = "train_" + global_s2v_npy
        else:
            sp = "test_" + global_s2v_npy        
        pkl_file = open(sp, 'rb')
        vec = pickle.load(pkl_file)
        pkl_file.close()
        return vec
        
