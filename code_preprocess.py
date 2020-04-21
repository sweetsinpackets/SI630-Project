import nltk
import re
import sys
import pandas as pd
import numpy as np
import string

global_tk_path = "tk.csv"

# this file creates a csv called "sub.csv"
def data_clean(path):
    data = pd.read_csv(path)

    #FIXME: For training process....
    #we never use some columns... drop for saving space
    data = data.drop(['id', 'grades'], axis=1)

    #FIXME: For predicting process it's the same because we use dev...

    # We create a new column saving the replaced headline
    data["replaced"] = ""
    # note that the . won't match /n, or need flags = re.DOTALL / re.S
    #   in train, test cases, no /n appears
    re_exp = "<.*/>"    # the regular expression of edited
    for (idx, line) in data.iterrows():   
        data["replaced"][idx] = re.sub(re_exp, line['edit'], line['original'])

    # if if_train:
    #     p = "train_" + global_preprocess_path
    # else:
    #     p = "test_" + global_preprocess_path
    # data.to_csv(global_preprocess_path, index=False, float_format='%.4f')
    return data


# this file reads the file "sub.csv" created by data_clean
# returns a dataframe
# in the following path, we will offer both read files method
# def data_read(preprocess_path):
#     return pd.read_csv(preprocess_path)


# preprocess data and tokenize, save to tk.csv
def data_preprocess(path, is_train):
    data = data_clean(path)

    # clean the samples with 0 score
    # data = data[ data['grades'] != 0.0]

    # for the following operations, we don't split the string for keeping invariant, hence it's better to remove or add some pre-processing

    # remove symbols
    data["original"] = data["original"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    data["replaced"] = data["replaced"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    data["original"] = data["original"].str.replace('[^\w\s]','')
    data["replaced"] = data["replaced"].str.replace('[^\w\s]','')

    # remove numbers
    data["original"] = data["original"].apply(lambda x: x.translate(str.maketrans("", "", string.digits)))
    data["replaced"] = data["replaced"].apply(lambda x: x.translate(str.maketrans("", "", string.digits)))

    # make all character lower case
    data["original"] = data["original"].apply(lambda x:x.lower())
    data['replaced'] = data["replaced"].apply(lambda x:x.lower())

    # remove nltk.stopwords
    data["original"] = data["original"].apply(lambda x:" ".join(x for x in x.split() if x not in nltk.corpus.stopwords.words("english")))
    data["replaced"] = data["replaced"].apply(lambda x:" ".join(x for x in x.split() if x not in nltk.corpus.stopwords.words("english")))

    # remove low term frequency words (only in replaced)
    word_freq = {}
    for (index, row) in data.iterrows():
        for i in row["replaced"].split():
            if i in word_freq:
                word_freq[i] = word_freq[i] + 1
            else:
                word_freq[i] = 1
        #from value higher to lower
    word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse = True)

        #slice the dictionary, pick last
    rare_threshold = -int(0.1*len(word_freq))
        #the sorted will turn it to a list of tuples, recover to dict
    rare_words = {word_freq[0]: word_freq[1] for k in word_freq[rare_threshold:]}

    data["replaced"] = data["replaced"].apply(lambda x:" ".join(x for x in x.split() if x not in rare_words))

    # tokenize to a new column
    data["tokenized"] = [nltk.word_tokenize(i) for i in data["replaced"]]

    # recover words to original form
    res = []
    for (index, row) in data.iterrows():
        res.append([nltk.stem.WordNetLemmatizer().lemmatize(w) for w in row["tokenized"]])
    data["tokenized"] = res

    # save to a csv
    if is_train:
        p = "train_" + global_tk_path
    else:
        p = "test_" + global_tk_path 

    data.to_csv(p, index=False, float_format='%.4f')
    return data



# a top level function to get a dataframe
def get_preprocessed_data(path, is_train, is_first_time=True):
    if is_first_time:
        return data_preprocess(path, is_train)
    else:
        if is_train:
            p = "train_" + global_tk_path
        else:
            p = "test_" + global_tk_path 
        return pd.read_csv(p)