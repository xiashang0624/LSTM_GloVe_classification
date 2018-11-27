import pandas as pd
import numpy as np
import sys, os, csv
from collections import defaultdict
from scipy.sparse import csr_matrix, csc_matrix, diags
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Activation, Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from gensim.parsing.porter import PorterStemmer
from sklearn.utils import class_weight
from keras import backend as K
from datetime import datetime

print ('Loading dataset...')
df = pd.read_csv('Accepted_answer_prediction_data_train.txt', sep ='\t',
                 header=None,names=['C_ID','Description','U_ID','Type','Time'])
label = pd.read_csv('Accepted_answer_prediction_labels_train.txt', sep ='\t',
                 header=None,names=['C_ID','label'])
df = pd.merge(df, label, on=['C_ID'])
embedding_file='glove.6B.100d.txt'

df['length'] = [len(str(des).split()) for des in df['Description'].tolist()]

emb_D = 100 # Dimension of the embedding.  Here we chose 100
doc_len = 300 # we chose certain number of words used in the model. For documents with fewer words, zero padding will be used.

print ('Tokenizing and Embedding...')
Train_Doc = df["Description"].fillna("NA").values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(Train_Doc))
Train_tokenize = tokenizer.texts_to_sequences(Train_Doc)
X_t = pad_sequences(Train_tokenize, maxlen=doc_len)
y = df['label'].values

p = PorterStemmer()
embedding = {}
f = open(embedding_file, encoding="utf8")
for word_vec in f:
    word_vec = word_vec.split()
    word = word_vec[0]
    vec = np.asarray(word_vec[1:])
    embedding[p.stem(word)] = vec.astype('float32')

print ('the length of the embedding volcabulary after stemming treatment is ', len(embedding))
emb_values = np.stack(embedding.values())
emb_mean = emb_values.mean()
emb_std = emb_values.std()
vocab = tokenizer.word_index
vocab_len = len(vocab)+1 # here the reason of adding 1 is because the index of word in vocab starts at 1
embed_Matrix = np.random.normal(emb_mean, emb_std, (vocab_len, emb_D))
for w, i in vocab.items():
    embed_Vector = embedding.get(w)
    if embed_Vector is not None:
        embed_Matrix[i] = embed_Vector
# save the tokenizer for testing
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

print ('Prepare model...')
a = Input(shape=(doc_len,))
b = Embedding(vocab_len, emb_D, weights=[embed_Matrix])(a)
b = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(b)
b = GlobalMaxPool1D()(b)
b = Dense(100, activation="elu")(b)
b = Dropout(0.2)(b)
b = Dense(50, activation="elu")(b)
b = Dropout(0.2)(b)
# b = Dense(20, activation="elu")(b)
# b = Dropout(0.15)(b)
b = Dense(1, activation="sigmoid")(b)
model = Model(inputs=a, outputs=b)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity, specificity, 'accuracy'])


print ('Model training...')
# 4-fold cross-validation
folds = 4
kfold = StratifiedKFold(n_splits=folds, shuffle=True)
iter_kfold = 0
validation_result = []
total_result = []
for train, test in kfold.split(X_t, y):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y[train]), y[train])
    model.fit(X_t[train], y[train], batch_size=32, epochs=20, validation_split=0.2,class_weight={0:class_weights[0],1:class_weights[1]});
    validation_result.append(model.evaluate(X_t[test], y[test], verbose=0))
    total_result.append(model.evaluate(X_t, y, verbose=0))
    model_name = 'model_' + str(iter_kfold) + '.h5'
    model.save(model_name)
    iter_kfold += 1

f=open('training_result.txt', 'w')
now = datetime.now()
f.write('Model trained at Timestamp: {:%Y-%m-%d %H:%M:%S}\n'.format(datetime.now()))
f.write('{}-fold validation\n'.format(folds))
for i in range(folds):
    f.write('model {0:.0f}: Recall={1:.3f} Selectivity={2:.3f} Accuracy={3:.3f}.\n'.format(i, validation_result[i][1], validation_result[i][2],validation_result[i][3]))
f.close()


