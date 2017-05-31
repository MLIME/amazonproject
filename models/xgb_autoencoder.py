#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.layers import Input, Dense, Dropout, Flatten, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras_metrics import KerasMetrics
from base_model import BaseModel
from utils import get_timestamp
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
from sklearn.multiclass import OneVsRestClassifier


class XGBEncoderModel(BaseModel):
    def __init__(self):
        self.sample_size = 5000
        
        self.label_indexes = dict()
        self.label_groups = dict()
        self.train_rows = dict()
        self.valid_rows = dict()

        self.label_groups[0] = ['agriculture']
        self.label_groups[1] = ['artisinal_mine']
        self.label_groups[2] = ['bare_ground']
        self.label_groups[3] = ['blooming']
        self.label_groups[4] = ['blow_down']
        self.label_groups[5] = ['clear']
        self.label_groups[6] = ['cloudy']
        self.label_groups[7] = ['conventional_mine']
        self.label_groups[8] = ['cultivation']
        self.label_groups[9] = ['habitation']
        self.label_groups[10] = ['haze']
        self.label_groups[11] = ['partly_cloudy']
        self.label_groups[12] = ['primary']
        self.label_groups[13] = ['road']
        self.label_groups[14] = ['selective_logging']
        self.label_groups[15] = ['slash_burn']
        self.label_groups[16] = ['water']

        self.label_group_train_rows = dict()
        self.label_group_valid_rows = dict()
        self.label_group_cols = dict()

        self.models = []
        self.result_cols = np.empty(0).astype('uint8')


    def args(self):
        '''
        Lists parameters
        '''
        pass


    def initialize(self, num_categories, args):
        '''
        Initialize parameters
        '''
        self.metrics = KerasMetrics()
        self.num_categories = num_categories

        self.base_dir = args.get('base_dir', '.')
        self.batch_size = args.get('batch_size', 8)
        self.image_multiplier = args.get('image_multiplier', 1)
        self.num_epochs = args.get('num_epochs', 1)
        self.use_generator = args.get('use_generator', False)
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

            stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')
            chkpt = ModelCheckpoint(chkpt_file_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

            self.callbacks = [stopper, chkpt]

            self.model, self.encoder = self._create_model(
                num_categories=self.num_categories,
                image_base_size=self.image_base_size,
                channels=self.channels,
                optimizer='nadam',
                init='he_normal', 
                window_size=7,
                hidden_layer_size=1024,
                activation='relu', 
                dropout1=0.2,
                dropout2=0.5)

            if self.use_generator:
                self._create_datagens()


        xgb = XGBClassifier(n_estimators=700, max_depth=7, learning_rate=0.1, silent=False)
        self.ovrmodel = OneVsRestClassifier(xgb, n_jobs=4)


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits model
        '''
        self.model.fit(X_train, X_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_data=(X_valid, X_valid),
            verbose=1, callbacks=self.callbacks)

        new_X_train = self.encoder.predict(X_train)
        new_X_valid = self.encoder.predict(X_valid)
        
        train_size = len(X_train)
        valid_size = len(X_valid)
        
        self.ovrmodel.fit(new_X_train.reshape(train_size, int((self.image_base_size//8)*(self.image_base_size//8)*self.channels)), y_train)
        y_pred = self.ovrmodel.predict(new_X_valid.reshape(valid_size, int((self.image_base_size//8)*(self.image_base_size//8)*self.channels)))
        print('Validation f2: %.2f' % fbeta_score(y_pred, y_valid, beta=2, average='samples'))


    def predict(self, X_test):
        '''
        Predicts labels using a fitted model
        '''

        new_X_test = self.encoder.predict(X_test)

        test_size = len(X_test)
        return self.model.predict(new_X_test.reshape(test_size, int((self.image_base_size//8)*(self.image_base_size//8)*self.channels)))


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
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.0,
            zoom_range=0.1,
            fill_mode='nearest')

        self.valid_datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.0,
            zoom_range=0.1,
            fill_mode='nearest')


    def _create_model(self, num_categories, image_base_size=256, channels=3, optimizer='adam', init='he_uniform', window_size=7, hidden_layer_size=256, activation='relu', dropout1=0.2, dropout2=0.5):
        '''
        Create a Keras CNN model structure
        '''

        input_img = Input(shape=(image_base_size, image_base_size, channels))

        model = Conv2D(32, (window_size, window_size), padding='same', activation=activation)(input_img)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Conv2D(16, (window_size, window_size), padding='same', activation=activation)(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Conv2D(8, (window_size, window_size), padding='same', activation=activation)(model)
        encoder = MaxPooling2D(pool_size=(2, 2))(model)

        model = Conv2D(8, (window_size, window_size), padding='same', activation=activation)(encoder)
        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(16, (window_size, window_size), padding='same', activation=activation)(model)
        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(32, (window_size, window_size), padding='same', activation=activation)(model)
        model = UpSampling2D(size=(2, 2))(model)
        #model = Conv2D(32, (window_size, window_size), padding='same', activation=activation)(model)
        #model = UpSampling2D(size=(2, 2))(model)
        decoder = Conv2D(3, (window_size, window_size), padding='same', activation='sigmoid')(model)

        autoencoder = Model(input_img, decoder)
        autoencoder.compile(loss='binary_crossentropy', optimizer=optimizer)
    
        return (autoencoder, encoder)

