from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, BatchNormalization, LSTM, TimeDistributed, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences

import vggish_params
import vggish_keras

import load_data

# Vggish weights
checkpoint_path = 'vggish_weights.ckpt'

# Path to data
data_path = "/cs/tmp/jm361/TestDatasetWild/"
data_output_path = "/cs/tmp/jm361/CRNNTestDatasetWild.txt"

timeSteps = 5
# Load the data
classes, features, labels = load_data.load_features_labels(data_output_path)
# Breaking features into timesteps
feats = []
timeStepLabels = []
for i, feature in enumerate(features):
    fts = np.array_split(feature, ceil(len(feature)/timeSteps))
    # Assigning a label for each new sequence
    for f in fts:
        timeStepLabels.append(labels[i])
    # Padding any short sequences with 0's
    fts = pad_sequences(fts, timeSteps)
    feats.append(fts)

# Correcting format of new labels
timeStepLabels = np.array(timeStepLabels)
# Adding all list of time steps from each sound file together
feats = np.concatenate(feats)

# Creating testing and training data, shuffles by default
X_train, X_test, y_train, y_test = train_test_split(feats, timeStepLabels, test_size=0.10)

# Load vggish model
vggish_model = vggish_keras.get_vggish_keras()
vggish_model.load_weights(checkpoint_path, by_name=True)
# Additions to model
model = Sequential()
model.add(TimeDistributed(vggish_model))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(len(classes)))
model.add(Dropout(0.3))
model.add(Activation('softmax'))

# Configure the model for training
adam = Adam(lr=vggish_params.LEARNING_RATE, epsilon=vggish_params.ADAM_EPSILON)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=14, validation_split=0.20)
model.summary()
print("Test")
result = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(result)