import pandas as pd
import numpy as np
import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scp
import random

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import statsmodels.api as sm

import tensorflow as tf

print(tf.__version__)

vaers_df = pd.read_csv(filepath_or_buffer='../../data/raw/VAERS/2021VAERSDATA.csv', sep=',', encoding = "ISO-8859-1", dtype={
    'RPT_DATE': str,
    'ER_VISIT': str,
    'V_FUNDBY': str,
    'SYMPTOM_TEXT': str
})

# Differentiate features and target

X = vaers_df[['SYMPTOM_TEXT']]
y = vaers_df[['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'DISABLE', 'BIRTH_DEFECT', 'OFC_VISIT', 'ER_ED_VISIT']].isna().T.all()

# Splitting train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
X_train = np.array(X_train).reshape(X_train.shape[0]).astype('str')
X_test = np.array(X_test).reshape(X_test.shape[0]).astype('str')

# Preprocess Texts

vocab_size = 40000
embedding_dim = 32
max_length = 200
trunc_type="post"
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# ML Model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test))
