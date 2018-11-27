# CS 498HS project

## Author: Xia Shang (xshang3@illinois.edu), Cheng Tao (ctao6@illinois.edu)
### Date: 11/27/2018



Here we build a LSTM classification model with Glovec embedding to predict whether a Stack Exchange answer will be accepted or not.


To use the pre-trained model, make sure you have the the following files in the current directory:
---

**model.h5** This file contains the pretrained weights for the LSTM DNN model.

**tokenizer.pickle** This file contains the embedding information and the tokenizer obtained during the  model training procss.


If you want to re-train the model, make sure you download the following files/packages.
---

**glove.6B.100d** This is a public embedding resource file and it can be obtained from https://nlp.stanford.edu/projects/glove/.

**Accepted_answer_prediction_data_train.txt** This is a given file from task 1 dataset in the class website.

**Accepted_answer_prediction_labels_train.txt** This is a given file from task 1 dataset in the class website.

To use the pre-trained model in testing cases (test.py)
---
To run 


