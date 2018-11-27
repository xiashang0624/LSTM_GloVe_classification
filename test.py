
#add following lines in train:
'''
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save('model.h5')
'''
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


test_file = input('Test filename:')
if not os.path.isfile(test_file):
    print('file not exist')
    sys.exit(0)
df = pd.read_csv(test_file, sep ='\t',
                 header=None,names=['C_ID','Description','U_ID','Type','Time'])
model = load_model('model.h5',custom_objects={'sensitivity': sensitivity,'specificity':specificity})

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

Test_tokenize = tokenizer.texts_to_sequences(df['Description'].fillna("NA").values)
X_t = pad_sequences(Test_tokenize, maxlen=300)
prediction = np.round(model.predict(X_t))
np.savetxt('output.txt',prediction.astype(int))
# Accepted_answer_prediction_data_train.txt

