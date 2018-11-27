# CS 498HS project

## Author: Xia Shang (xshang3@illinois.edu), Cheng Tao (ctao6@illinois.edu)
### Date: 11/27/2018



Here we built a LSTM classification model with Glovec embedding to predict whether a Stack Exchange answer will be accepted or not.


To use the pre-trained model, make sure you have the the following files in the current directory:
---

**model.h5** This file contains the pretrained weights for the LSTM DNN model.

**tokenizer.pickle** This file contains the embedding information and the tokenizer obtained during the  model training procss.

To use the pre-trained model in testing cases (test.py)
---
To run the test.py model....


If you want to re-train the model, make sure you download the following files/packages.
---

**glove.6B.100d** This is a public embedding resource file and it can be obtained from https://nlp.stanford.edu/projects/glove/.

**Accepted_answer_prediction_data_train.txt** This is a given file from task 1 dataset in the class website.

**Accepted_answer_prediction_labels_train.txt** This is a given file from task 1 dataset in the class website.

To re-train the model, type ```python train.py```.
The entire training process may take up to 3 hours, depending on your computing power.

Since 4-fold cross-validation is used, 4 different models will be trained and evaluated. A "training_result.txt" will be also generated once the training process is done.  Since this is a binary classification problem, the modeling results will be evaluated in the following three categories: recall (sensitivity, true positive rate), specivity (true negative rate), and accuracy.

An example of the "training_result.txt" is shown as follows:

```
Model trained at Timestamp: 2018-11-27 12:37:49
4-fold validation
model 0: Recall=0.819 Selectivity=0.983 Accuracy=0.947.
model 1: Recall=0.865 Selectivity=0.954 Accuracy=0.936.
model 2: Recall=0.841 Selectivity=0.968 Accuracy=0.938.
model 3: Recall=0.833 Selectivity=0.978 Accuracy=0.949.
```
In addition, the same training process is also available in the jupyter notebook **LSTM.ipynb**.
