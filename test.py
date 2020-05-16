import pandas as pd
import numpy as np

from keras.models import load_model

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
s = "lets meet around 12 tommorrow"

test_x = [[word2idx[w] for w in s.split(" ")]]
test_x = pad_sequences(maxlen=max_len, sequences=test_x, padding="post", value=n_words - 1)

result = model.predict(np.array([test_x[0]]))
result = np.argmax(result, axis=-1)

print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(test_x[0], result[0]):
    print("{:15}: {}".format(words[w], tags[pred]))


