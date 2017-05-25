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

class KerasMultiCNNModel(BaseModel):
    def __init__(self):
        self.sample_size = 1000
        
        self.label_indexes = dict()
        self.label_groups = dict()
        self.train_rows = dict()
        self.valid_rows = dict()

        self.label_groups[0] = ['bare_ground', 'primary']
        self.label_groups[1] = ['clear', 'cloudy', 'haze', 'partly_cloudy']
        self.label_groups[2] = ['blooming', 'blow_down', 'slash_burn']
        self.label_groups[3] = ['road', 'selective_logging', 'water']
        self.label_groups[4] = ['artisinal_mine', 'conventional_mine', 'habitation']
        self.label_groups[5] = ['agriculture', 'cultivation']

        self.label_group_train_rows = dict()
        self.label_group_valid_rows = dict()
        self.label_group_cols = dict()

        self.models = []
        self.result_cols = np.empty(0).astype('uint8')


    def args(self):
        '''
        Lists KerasCNN parameters
        '''
        pass


    def initialize(self, num_categories, args):
        '''
        Initialize Keras CNN parameters
        '''
        metrics = KerasMetrics()
        num_categories = num_categories

        self.backend = args.get('--backend', 'tf')

        self.base_dir = args.get('base_dir', '.')
        self.batch_size = args.get('batch_size', 8)
        self.image_multiplier = args.get('image_multiplier', 1)
        self.num_epochs = args.get('num_epochs', 1)
        self.use_generator = args.get('use_generator', False)
        self.label_names = args.get('label_names', None)
        
        saved_model_name = args.get('saved_model_name', None)
        image_base_size = args.get('image_base_size', 256)
        channels = args.get('channels', 3)

        if saved_model_name:
        	self.model = load_model(saved_model_name, custom_objects={'f2_score': metrics.f2_score})
        else:
            chkpt_file_name = os.path.join(self.base_dir, get_timestamp() + '_chkpt_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
            model_file_name = os.path.join(self.base_dir, get_timestamp() + '_final_model.h5')

            stopper = EarlyStopping(monitor='val_f2_score', min_delta=0.0001, patience=2, verbose=1, mode='auto')
            chkpt = ModelCheckpoint(chkpt_file_name, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

            self.callbacks = [stopper, chkpt]

            if self.use_generator:
                self._create_datagens()


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a Keras CNN Model
        '''
        for label in self.label_names:
            self.label_indexes[label] = np.where(self.label_names == label)[0][0]
            self.train_rows[label] = np.random.choice(np.where(y_train[:, self.label_indexes[label]] == 1)[0], self.sample_size)
            self.valid_rows[label] = np.random.choice(np.where(y_valid[:, self.label_indexes[label]] == 1)[0], self.sample_size)

        for group, labels in label_groups.items():
            train_rows = np.empty(0).astype('uint8')
            valid_rows = np.empty(0).astype('uint8')
            cols = np.empty(0).astype('uint8')
    
            for label in labels:
                cols = np.hstack((cols, label_indexes[label]))
                train_rows = np.hstack((rows, self.train_rows[label]))
                valid_rows = np.hstack((rows, self.valid_rows[label]))

            self.label_group_cols[group] = np.unique(cols)
            self.label_group_train_rows[group] = np.unique(train_rows)
            self.label_group_valid_rows[group] = np.unique(valid_rows)
        
        
        for i in range(len(self.label_groups)):
            train_rows = self.label_group_train_rows[i]
            valid_rows = self.label_group_valid_rows[i]
            cols = label_group_cols[i]

            self.result_cols = np.hstack((result_cols, cols))

            new_X_train = X_train[train_rows, :]
            new_y_train = y_train[train_rows, cols]
            new_X_valid = X_valid[valid_rows, :]
            new_y_valid = y_valid[valid_rows, cols])

            model = self._create_model(
                num_categories=len(cols),
                f2_score=metrics.f2_score, 
                image_base_size=image_base_size,
                channels=channels,
                optimizer='nadam',
                init='he_normal', 
                window_size=7,
                hidden_layer_size=1024,
                activation='relu', 
                dropout1=0.2,
                dropout2=0.5)

            if self.use_generator:
                self.train_datagen.fit(new_X_train)
                self.valid_datagen.fit(new_X_valid)

                self.model.fit_generator(
                    self.train_datagen.flow(new_X_train, new_y_train, batch_size=self.batch_size * 2),
                    steps_per_epoch=(len(new_X_train) / self.batch_size) * self.image_multiplier,
                    epochs=self.num_epochs,
                    validation_data=self.train_datagen.flow(new_X_valid, new_y_valid, batch_size=self.batch_size * 2),
                    validation_steps=len(new_X_valid) / self.batch_size,
                    verbose=1, callbacks=self.callbacks)
            else:
                model.fit(new_X_train, new_y_train,
                    batch_size=self.batch_size,
                    epochs=self.num_epochs,
                    validation_data=(new_X_valid, new_y_valid),
                    verbose=1, callbacks=self.callbacks)

            self.models.append(model)


    def predict(self, X_test):
        '''
        Predicts labels using a fitted Keras CNN Model
        '''
        y_pred = np.empty(0).astype('uint8')

        for i in range(len(self.label_groups)):
            y_pred = np.hstack((y_pred, self.models[i].predict(X_test)))


        new_cols = np.hstack(([np.where(self.result_cols == i) for i in np.arange(len(self.result_cols))])).squeeze()
        return y_pred[:,new_cols]


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
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.0,
            zoom_range=0.3,
            fill_mode='nearest')

        self.valid_datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.0,
            zoom_range=0.3,
            fill_mode='nearest')


    def _create_model(self, num_categories, f2_score, image_base_size=256, channels=3, optimizer='adam', init='he_uniform', window_size=7, hidden_layer_size=256, activation='relu', dropout1=0.2, dropout2=0.5):
        '''
        Create a Keras CNN model structure
        '''

        model = Sequential()

        model.add(Conv2D(96, (window_size, window_size), input_shape=(image_base_size, image_base_size, channels), padding='same', activation=activation, kernel_initializer = init))
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
        model.add(Dense(num_categories, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[f2_score])
    
        return model
