#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
from sklearn.multiclass import OneVsRestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras_metrics import KerasMetrics
from base_model import BaseModel
from utils import get_timestamp


class KerasSKOneVsAllModel(BaseModel):
    def args(self):
        '''
        Lists model parameters
        '''
        pass


    def initialize(self, num_categories, args):
        '''
        Initialize model parameters
        '''
        self.metrics = KerasMetrics()
        self.num_categories = num_categories

        self.base_dir = args.get('base_dir', '.')
        self.batch_size = args.get('batch_size', 8)
        self.num_epochs = args.get('num_epochs', 1)
        self.label_names = args.get('label_names', None)
        
        saved_model_name = args.get('saved_model_name', None)
        self.image_base_size = args.get('image_base_size', 256)
        self.channels = args.get('channels', 3)

        self.timestamp = get_timestamp()

        if saved_model_name:
        	self.model = load_model(saved_model_name, custom_objects={'f2_score': metrics.f2_score})
        else:
            chkpt_file_name = os.path.join(self.base_dir, self.timestamp + '_' + self.__class__.__name__ + '_chkpt_weights.{epoch:02d}-{val_f2_score:.2f}.hdf5')
            model_file_name = os.path.join(self.base_dir, self.timestamp + '_' + self.__class__.__name__ +  '_final_model.h5')

            stopper = EarlyStopping(monitor='val_f2_score', min_delta=0.0001, patience=2, verbose=1, mode='auto')
            chkpt = ModelCheckpoint(chkpt_file_name, monitor='val_f2_score', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

            self.callbacks = [stopper, chkpt]


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a model
        '''

        k = KerasClassifier(build_fn=self._create_model(
                num_categories=self.num_categories,
                f2_score=self.metrics.f2_score,
                image_base_size=self.image_base_size,
                channels=X_train.shape[3],
                optimizer='nadam',
                init='he_normal', 
                window_size=7,
                hidden_layer_size=512,
                activation='relu', 
                dropout1=0.2,
                dropout2=0.5))

        self.model = OneVsRestClassifier(estimator=k, n_jobs=1)
        self.model.set_params(estimator__batch_size=self.batch_size, estimator__epochs=self.num_epochs, estimator__verbose=1, estimator__validation_split=0.2)
        self.model.fit(X_train, y_train) 


    def predict(self, X_test):
        '''
        Predicts labels using a fitted model
        '''
        return self.model.predict(X_test)


    def _create_model(self, num_categories, f2_score, image_base_size=256, channels=3, optimizer='adam', init='he_uniform', window_size=7, hidden_layer_size=256, activation='relu', dropout1=0.2, dropout2=0.5):
        '''
        Create a Keras CNN model structure
        '''

        model = Sequential()

        model.add(BatchNormalization(input_shape=(image_base_size, image_base_size, channels)))
        model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout1))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout1))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout2))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout2))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (window_size, window_size), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(num_categories, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[f2_score])
    
        return model
