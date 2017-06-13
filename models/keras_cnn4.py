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
from keras.preprocessing.image import ImageDataGenerator
from keras_metrics import KerasMetrics
from base_model import BaseModel
from utils import get_timestamp


class KerasCNNModel4(BaseModel):
    def __init__(self):
        pass


    def args(self):
        '''
        Lists KerasCNN parameters
        '''
        pass


    def initialize(self, num_categories, args):
        '''
        Initialize Keras CNN parameters
        '''
        self.metrics = KerasMetrics()
        num_categories = num_categories

        self.backend = args.get('--backend', 'tf')

        self.timestamp = get_timestamp()

        self.base_dir = args.get('base_dir', '.')
        self.batch_size = args.get('batch_size', 8)
        self.image_multiplier = args.get('image_multiplier', 1)
        self.num_epochs = args.get('num_epochs', 1)
        self.use_generator = args.get('use_generator', False)
        
        saved_model_name = args.get('saved_model_name', None)
        image_base_size = args.get('image_base_size', 256)
        channels = args.get('channels', 3)
        
        if saved_model_name:
            self.model = load_model(saved_model_name, custom_objects={'f2_score': self.metrics.f2_score})
        else:
            self.chkpt_file_name = os.path.join(self.base_dir, self.timestamp + '_' + self.__class__.__name__ + '_chkpt_weights.hdf5')
            model_file_name = os.path.join(self.base_dir, self.timestamp + '_' + self.__class__.__name__ +  '_final_model.h5')

            stopper = EarlyStopping(monitor='val_f2_score', min_delta=0.00005, patience=30, verbose=1, mode='max')
            chkpt = ModelCheckpoint(self.chkpt_file_name, monitor='val_f2_score', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

            self.callbacks = [stopper, chkpt]

            self.model = self._create_model(
                num_categories=num_categories,
                f2_score=self.metrics.f2_score, 
                image_base_size=image_base_size,
                channels=channels,
                optimizer='adam',
                init='he_normal', 
                window_size=3,
                hidden_layer_size=4096,
                activation='relu', 
                dropout1=0.2,
                dropout2=0.5)

            if self.use_generator:
                self._create_datagens()


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a Keras CNN Model
        '''
        
        if X_train.shape[2] > 4:
            self.use_generator = False

        if self.use_generator:
            self.train_datagen.fit(X_train)
            self.valid_datagen.fit(X_valid)

            self.model.fit_generator(
                self.train_datagen.flow(X_train, y_train, batch_size=self.batch_size * 2),
                steps_per_epoch=(len(X_train) / self.batch_size) * self.image_multiplier,
                epochs=self.num_epochs,
                validation_data=(X_valid, y_valid),
                validation_steps=len(X_valid) / self.batch_size,
                verbose=1, callbacks=self.callbacks)
        else:
            self.model.fit(X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.num_epochs,
                validation_data=(X_valid, y_valid),
                verbose=1, callbacks=self.callbacks)

        self.model = load_model(self.chkpt_file_name, custom_objects={'f2_score': self.metrics.f2_score})


    def predict(self, X_test):
        '''
        Predicts labels using a fitted Keras CNN Model
        '''
        
        return self.model.predict(X_test)


    def _create_datagens(self):
        '''
        Create image generators for data augmentation
        '''
        
        self.train_datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=90,
            width_shift_range=0.5,
            height_shift_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.3,
            zoom_range=0.3,
            fill_mode='nearest')

        self.valid_datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=90,
            width_shift_range=0.5,
            height_shift_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.3,
            zoom_range=0.3,
            fill_mode='nearest')


    def _create_model(self, num_categories, f2_score, image_base_size=256, channels=3, optimizer='adam', init='he_uniform', window_size=7, hidden_layer_size=256, activation='relu', dropout1=0.2, dropout2=0.5):
        '''
        Create a Keras CNN model structure
        '''

        model = Sequential()

        model.add(BatchNormalization(input_shape=(image_base_size, image_base_size, channels)))
        model.add(Conv2D(96, (11, 11), padding='same', activation=activation))
        model.add(Conv2D(96, (11, 11), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (5, 5), padding='same', activation=activation))
        model.add(Conv2D(256, (5, 5), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(384, (3, 3), padding='same', activation=activation))
        model.add(Conv2D(384, (3, 3), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding='same', activation=activation))
        model.add(Conv2D(256, (3, 3), padding='same', activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dense(num_categories, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[f2_score])
    
        return model
