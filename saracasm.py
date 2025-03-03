import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

!pip install --upgrade pip
!pip install seaborn plotly
!pip install transformers
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


#Configuring TPU's
#For this version of Notebook we will be using TPU's as we have to built a BERT Model

# !pip install --upgrade tensorflow
import tensorflow as tf
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    import tensorflow as tf
    import os
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(gpu)
#     strategy = tf.distribute.get_strategy()
    strategy = tf.distribute.MirroredStrategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


trainall = pd.read_csv('/kaggle/input/hindi-english-codemix-balanced/train.csv')
validation = pd.read_csv('/kaggle/input/hindi-english-codemix-balanced/valid.csv')
test = pd.read_csv('/kaggle/input/hindi-english-codemix-balanced/test.csv')
trainall.columns

columns = ['tweets', 'labels']
train = trainall = trainall[columns]
validation = validation[columns]
test = test[columns]
target = 'labels'

print(trainall.columns, test.columns)
print(trainall[target].unique())
print(trainall.shape, validation.shape, test.shape)

print(train['tweets'].apply(lambda x:len(str(x).split())).max())
print(validation['tweets'].apply(lambda x:len(str(x).split())).max())
print(test['tweets'].apply(lambda x:len(str(x).split())).max())


Data Preparation

def split_ds(ds):
    return ds[columns[0]], ds[columns[1]]
xtrain, ytrain = split_ds(trainall)
xvalid, yvalid = split_ds(validation)
xtest, ytest = split_ds(test)
scores_model = []
all_scores = []
xtrain.shape, xvalid.shape, xtest.shape


batch_size = 32
steps_per_epoch = 20
steps_per_epoch = np.ceil(len(xtrain)/(batch_size*strategy.num_replicas_in_sync))
print(xtrain.shape, ytrain.shape, xtest.shape)
print(steps_per_epoch)
print(strategy.num_replicas_in_sync)

# if ytrain
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def roc_auc_multi(predictions, target):
    # Compute ROC AUC for each class
    roc_auc_scores = []
    for i in range(target.shape[1]):
        roc_auc_score = roc_auc(predictions[:, i], target[:, i])
        roc_auc_scores.append(roc_auc_score)

    # Average ROC AUC scores
    avg_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
    return avg_roc_auc

def accuracy(predictions, target, threshold=0.5):
    pred = predictions.copy()
    pred[pred<=threshold] = 0
    pred[pred>threshold] = 1
    return metrics.accuracy_score(pred, target)*100

def f1_score(predictions, target, threshold=0.5):
    pred = predictions.copy()
    pred[pred<=threshold] = 0
    pred[pred>threshold] = 1
    return metrics.f1_score(pred, target)

def best_threshold_f1(predictions, target):
    # Calculate F1 scores for different threshold values
    threshold_values = np.arange(0,1,0.001)  # Use the thresholds obtained from the ROC curve
    f1_scores = [f1_score(predictions, target, threshold) for threshold in threshold_values]

    # Find the threshold that maximizes the F1 score
    best_threshold = threshold_values[np.argmax(f1_scores)]
    best_f1_score = max(f1_scores)

    print("Best Threshold:", best_threshold)
    print("Best F1 Score:", best_f1_score)
    return best_threshold, best_f1_score


def performance(predictions, target, threshold=None):
    if not threshold:
        threshold, f1 = best_threshold_f1(predictions, target)
    else:
        f1 = f1_score(predictions, target, threshold)        
    roc = roc_auc(predictions, target)
    acc = accuracy(predictions, target, threshold)

    print("Roc-Auc : %.2f" % (roc))
    print("Accuracy: %.2f%%" % (acc))
    print("F1-Score: %.2f" % (f1))
    return roc, acc, f1, threshold


# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)
xtest_seq = token.texts_to_sequences(xtest)


#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

word_index = token.word_index


%%time
with strategy.scope():
    # A simpleRNN without any pretrained embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()

%%time
model.fit(xtrain_pad, ytrain, steps_per_epoch=steps_per_epoch, epochs=5, validation_data=[xvalid_pad, yvalid], batch_size=batch_size*strategy.num_replicas_in_sync) #Multiplying by Strategy to run on TPU's


scores = model.predict(xvalid_pad)
performance(scores, yvalid)

scores_model.append({'Model': 'SimpleRNN','AUC_Score': roc_auc(scores,yvalid)})
all_scores.append(('RNN',*performance(scores, yvalid)),83)


Predicting if Tweets from users is a Sarcasm or Not

%%time
import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

pretrained_model = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
best_threshold = 0.564
    
def preprocess_tweets(tweets):
    input_ids = []
    attention_masks = []
    for tweet in tweets:
        encoded = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, return_tensors='tf')
        input_ids.append(tf.squeeze(encoded['input_ids']))
        attention_masks.append(tf.squeeze(encoded['attention_mask']))
    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_masks)



def get_model(model_dir='/kaggle/input/kj-twitter/bert-base-multilingual-uncased-LSTM.h5'):
    with strategy.scope():
        bert_model = TFBertModel.from_pretrained(pretrained_model)
        input_ids = Input(shape=(128,), dtype=tf.int32)
        attention_mask = Input(shape=(128,), dtype=tf.int32)
        bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
        x = lstm_output = LSTM(64, return_sequences=False)(bert_output)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[input_ids, attention_mask], outputs=output)
        model.load_weights(model_dir)

        model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = get_model('/kaggle/input/kj-twitter/bert-base-multilingual-uncased-LSTM.h5')    
def predict_tweets(tweets):
    test_input_ids, test_attention_masks = preprocess_tweets(tweets)
    scores = model.predict([test_input_ids, test_attention_masks])
#     print(scores)
    scores = map(lambda x:'sarcasm' if x>best_threshold else 'non-sarcasm', scores)
    df = pd.DataFrame({'tweets':tweets, 'labels':scores})
    return df

%%time
tweets = [
    'happy birthday',
    'happy birthday agar tum accha party diya to',
]

result = predict_tweets(tweets)
result

