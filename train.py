import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split

import pickle
import logging

from keras.models import load_model

FILE_PATH = "data/Training-data.csv"
PRE_PROCESSED_DATA = "data/processed_data.p"
MODEL_PATH = "data/model.h5"
TRAIN_INFORMATION = "data/info_testing.p"
TRAIN_TEST_SPLIT = 0.1
DATA_LIMIT = 100
OUTPUT_DIMENSIONS = 50
LSTM_UNITS = 100
DROP_OUT = 0.1


def load_dataset(FILE_PATH):
    """

    :param FILE_PATH:
    :return:
    """
    try:
        data = pd.read_csv("data/Training-data.csv")
    except Exception as error:
        logging.error("File Not found %s", error)
    else:
        logging.info("Data loaded successfully")

    return data


def pre_process(data, limit=None):
    """

    :param data:
    :return:
    """
    df = pd.DataFrame()
    df['Sentence #'] = ""
    df['Word'] = ""
    df["Tag"] = ""

    for row, sentence in data.iterrows():
        if limit is not None:
            if row == limit:
                return df

        if row % 100 == 0:
            print("Done "+str(row)+" of "+str(len(data)))

        food_type = sentence['food_type']
        time = sentence['time']
        date = sentence['date']
        location = sentence['location']

        query = str(sentence['query'])
        for word in query.split(' '):

            tag = 'O'
            if word == food_type or word == time or word == date or word == location:
                tag = 'food_type' if food_type == word else tag
                tag = 'time' if time == word else tag
                tag = 'date' if date == word else tag
                tag = 'location' if location == word else tag

            df = df.append({'Sentence #': 'Sentence: ' + str(row),
                            'Word': word,
                            'Tag': tag},
                           ignore_index=True)

    return df


def get_words_tags(data):
    """

    :param data:
    :return:
    """
    words = list(set(data["Word"].values))
    words.append("ENDPAD")

    tags = list(set(data["Tag"].values))

    return words, tags


class SentenceGetter(object):

    def __init__(self, data):
        """

        :param data:
        """
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        """

        :return:
        """
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def pred2label(pred, idx2tag):
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


def create_model():
    """

    :return:
    """
    logging.info("Creating LSTM model")
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words, output_dim=OUTPUT_DIMENSIONS, input_length=max_len)(input)  # 50-dim embedding
    model = Dropout(DROP_OUT)(model)
    model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=DROP_OUT))(
        model)  # variational biLSTM
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

    model = Model(input, out)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train):
    """

    :param model:
    :param X_train:
    :param y_train:
    :return:
    """
    logging.info("Training model..")
    model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.1, verbose=1)
    logging.info("Saving model to model.h5 file")

    try:
        model.save(MODEL_PATH)
    except Exception as error:
        logging.info("Error occured while saving the model")
    else:
        logging.info("Model saved successfully")

    return model


logging.info("Loading Dataset..")
dataset = load_dataset(FILE_PATH)

try:
    data = pickle.load(open(PRE_PROCESSED_DATA, "rb"))
except Exception as error:
    logging.error("Preprocessed file not found, try uncommenting the pre-processing function")

logging.info("Preprocessing...")
# Uncomment here to enable pre-processing
data = pre_process(dataset)
pickle.dump(data, open(PRE_PROCESSED_DATA, "wb"))

data["POS"] = "None"
words, tags = get_words_tags(data)

n_words = len(words)
logging.info("Number of unique words %d", n_words)
n_tags = len(tags)
logging.info("Number of unique tags %d", n_tags)

getter = SentenceGetter(data)
sentences = getter.sentences

max_len = OUTPUT_DIMENSIONS
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

logging.info("Creating embedding for words..")
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

logging.info("Creating embedding for tags..")
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT)

model = create_model()
model = train_model(model, X_train, y_train)
model = load_model(MODEL_PATH)

idx2tag = {i: w for w, i in tag2idx.items()}

info_testing = dict()
info_testing['idx2tag'] = idx2tag
info_testing['word2idx'] = word2idx
info_testing['tag2idx'] = tag2idx
info_testing['words'] = words
info_testing['tags'] = tags
info_testing['max_len'] = max_len
info_testing['n_words'] = n_words
pickle.dump(info_testing, open(TRAIN_INFORMATION, "wb"))

test_pred = model.predict(X_test, verbose=1)
pred_labels = pred2label(test_pred, idx2tag)
test_labels = pred2label(y_test, idx2tag)
logging.info("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
