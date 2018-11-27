
print ('importing libraries...')
from keras import backend as K
import pickle
import numpy as np
from keras.models import load_model
import pandas as pd
import os
from keras.preprocessing.sequence import pad_sequences

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


test_file = input('Type Test filename:')
if not os.path.isfile(test_file):
    print('file not exist')
    sys.exit(0)
df = pd.read_csv(test_file, sep ='\t',
                 header=None,names=['C_ID','Description','U_ID','Type','Time'])
print ('test file has been found, loading pre-trained model...')
model = load_model('model.h5',custom_objects={'sensitivity': sensitivity,'specificity':specificity})

print ('pre-trained model has been loaded, loading pre-trained tokenizer...')
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

Test_tokenize = tokenizer.texts_to_sequences(df['Description'].fillna("NA").values)
X_t = pad_sequences(Test_tokenize, maxlen=300)
print ('embedding is done, predicting outputs...')
prediction = np.round(model.predict(X_t))
print ('Write outputs to file...')
np.savetxt('output.txt',prediction.astype('int32'))
print ('File was saved to output.txt...')
# Accepted_answer_prediction_data_train.txt

