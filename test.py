import numpy as np
import pickle

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MODEL_PATH = "model.h5"
TRAIN_INFORMATION = "info_testing.p"

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

try:
    print("Loading NER model...")
    model = load_model(MODEL_PATH, compile=True)
except Exception as error:
    print("Error while loading model, does the model exist ?")
else:
    print("Model loaded successfully")

try:
    print("Loading information used while training")
    info_testing = pickle.load(open(TRAIN_INFORMATION, "rb"))
    idx2tag = info_testing['idx2tag']
    word2idx = info_testing['word2idx']
    tag2idx = info_testing['tag2idx']
    words = info_testing['words']
    tags = info_testing['tags']
    max_len = info_testing['max_len']
    n_words = info_testing['n_words']
except Exception as error:
    print("Error while loading information, does the file exist ?")
else:
    print("Information loaded successfully")

while True:
    try:
        print ("Enter sentence for NER..")
        sentence = raw_input()
        for word in sentence.split(" "):
            if word2idx.get(word) is None:
                print("Word not found ", word)
                print("Try again..")
                break

        else:

            test_x = [[word2idx[w] for w in sentence.split(" ")]]
            test_x = pad_sequences(maxlen=max_len, sequences=test_x, padding="post", value=n_words - 1)

            result = model.predict(np.array([test_x[0]]))
            result = np.argmax(result, axis=-1)

            print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
            for w, pred in zip(test_x[0], result[0]):
                if words[w] == 'ENDPAD':
                    break
                print("{:15}: {}".format(words[w], tags[pred]))

    except Exception as error:
        print("Some Error occurred %s", error)
