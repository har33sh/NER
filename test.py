import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import pickle

data = pd.read_csv("Training-data.csv")
data.tail(10)

df = pd.DataFrame()
df['Sentence #'] = ""
df['Word'] = ""
df["Tag"] = ""

for row, sentence in data.iterrows():
    if row % 100 == 0:
        print(row)
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


# df = process_data(data)

df["POS"] = "None"
data = df

words = list(set(data["Word"].values))
words.append("ENDPAD")

n_words = len(words);
n_words

tags = list(set(data["Tag"].values))

n_tags = len(tags);
n_tags

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)


sent = getter.get_next()

print(sent)

sentences = getter.sentences

plt.style.use("ggplot")

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

tags = list(set(data["Tag"].values))

max_len = 50
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

#
# word2idx["tommorrow"]
#
# tag2idx["date"]
#
# tag2idx['time']

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)


y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

y = [to_categorical(i, num_classes=n_tags) for i in y]


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)  # 50-dim embedding
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

# In[45]:


model = Model(input, out)

# In[46]:


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# In[47]:


history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

# In[48]:


hist = pd.DataFrame(history.history)

# In[49]:


hist

# In[50]:


plt.figure(figsize=(12, 12))
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.show()

# In[51]:


i = 0
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(X_te[i], p[0]):
    print("{:15}: {}".format(words[w], tags[pred]))

# In[59]:


X_test = X_te
y_test = y_te

idx2tag = {i: w for w, i in tag2idx.items()}


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


test_pred = model.predict(X_test, verbose=1)
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)

# In[60]:


print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

# In[61]:


# In[ ]:




