#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

np.random.seed(777)

image_base_size = 96

label_file_name = 'train.csv'
label_file = pd.read_csv(label_file_name)
label_file = label_file.sort_values('image_name')

vec = CountVectorizer(min_df=1)
labels = vec.fit_transform(label_file['tags'])

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

num_categories = y_train.shape[1]

def create_model(optimizer='adam', init='he_uniform', window_size=7, hidden_layer_size=32, activation='relu', dropout1=0.2, dropout2=0.5):
    model = Sequential()

    model.add(Conv2D(32, (window_size, window_size), padding='same', activation=activation, input_shape=(image_base_size, image_base_size, 4), kernel_initializer = init))
    model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout1))
    model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout1))
    model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(hidden_layer_size, activation=activation))
    model.add(Dense(hidden_layer_size, activation=activation))
    model.add(Dense(hidden_layer_size, activation=activation))
    model.add(Dense(hidden_layer_size, activation=activation))
    model.add(Dense(hidden_layer_size, activation=activation))
    model.add(Dense(hidden_layer_size, activation=activation))
    model.add(Dropout(dropout2))
    model.add(Dense(num_categories, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def get_labels(y):
    y[y >= 0.5] = True
    y[y < 0.5] = False
    return np.array(vec.inverse_transform(y))


if os.path.exists('final_model.h5'):
	saved_model = load_model('final_model.h5')


if not saved_model:
	stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=3, verbose=1, mode='auto')

	callbacks = []
	callbacks.append(stopper)

	model = KerasClassifier(build_fn=create_model, verbose=1)

	optimizers = ['adam']
	inits = ['he_uniform']
	epochs = [50]
	batch_sizes = [24]
	window_sizes = [7]
	hidden_layer_sizes = [64]
	dropouts1 = [0.2]
	dropouts2 = [0.5]
	activations = ['relu']

	param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batch_sizes, 
					  init=inits, window_size=window_sizes, 
					  hidden_layer_size=hidden_layer_sizes, 
					  activation=activations,
					  dropout1=dropouts1,
					  dropout2=dropouts2)

	fit_params = dict(validation_split=0.3,
					  verbose=1,
					  callbacks=callbacks)

	grid = GridSearchCV(estimator=model, param_grid=param_grid, fit_params=fit_params, cv=3, verbose=1)
	grid_cnn = grid.fit(X_train, y_train)

	print("Best: %f using %s" % (grid_cnn.best_score_, grid_cnn.best_params_))

	final_model = grid_cnn.best_estimator_
	final_model.model.save('final_model.h5')
else:
	final_model = saved_model

print('Predicting on training data')
y_pred = final_model.predict_proba(X_train)

pred_labels = get_labels(y_pred)
train_labels = get_labels(y_train)

pred_labels_list = []
train_labels_list = []

for i in range(len(y_pred)):
	print(label_file['image_name'][i] + ': ' + ','.join(train_labels[i]) + '|' + ','.join(pred_labels[i]))
	pred_labels_list.append(','.join(pred_labels[i]))
	train_labels_list.append(','.join(train_labels[i]))

label_file['train_labels'] = train_labels_list
label_file['pred_labels'] = pred_labels_list

label_file.to_csv('train_result.csv')

X_test = np.load('X_test.npy')
test_img_names = pd.read_csv('test_img_names.csv')

n = 50

print('Predicting on test data')
y_pred = final_model.predict_proba(X_test[0:n,:])
pred_labels = get_labels(y_pred)

for i in range(n):
    print(test_img_names['image_name'][i] + ': ' + ','.join(pred_labels[i]))

