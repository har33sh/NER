import numpy as np
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

FILE_PATH = "Training-data.csv"

def pred2label(pred):
    """

    :param pred:
    :return:
    """
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

model = load_model("model.h5")

info_testing = pickle.load(open( "info_testing.p", "rb"))
idx2tag = info_testing['idx2tag']
word2idx = info_testing['word2idx']
tag2idx = info_testing['tag2idx']
words = info_testing['words']
tags = info_testing['tags']
max_len = info_testing['max_len']
n_words = info_testing['n_words']

while True:
    try:
        print ("Enter sentence for NER..")
        sentence = raw_input()
        test_x = [[word2idx[w] for w in sentence.split(" ")]]
        print(test_x)
        # test_x = []
        # for w in sentence.split(" "):
        #     x = []
        #     if word2idx.get(w):
        #         x.append(word2idx.get(w))
        #     else:
        #         x.append(word2idx.get("."))
        #     test_x = [x]

        test_x = pad_sequences(maxlen=max_len, sequences=test_x, padding="post", value=n_words - 1)
        print("chech")
        result = model.predict(np.array([test_x[0]]))
        print("dfgd")
        result = np.argmax(result, axis=-1)

        print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
        for w, pred in zip(test_x[0], result[0]):
            if words[w] == 'ENDPAD':
                break
            print("{:15}: {}".format(words[w], tags[pred]))
    except Exception as error:
        print("Error %s", error)

