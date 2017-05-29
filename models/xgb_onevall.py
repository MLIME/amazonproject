#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from base_model import BaseModel
from utils import get_timestamp
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
from sklearn.multiclass import OneVsRestClassifier

class XGBModel(BaseModel):
    def __init__(self):
        pass


    def args(self):
        '''
        Lists parameters
        '''
        pass


    def initialize(self, num_categories, args):
        '''
        Initialize parameters
        '''
        self.num_categories = num_categories
        self.timestamp = get_timestamp()

        self.image_base_size = args.get('image_base_size', 64)
        self.channels = args.get('channels', 3)

        xgb = XGBClassifier()
        self.model = OneVsRestClassifier(xgb)


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a model
        '''
        
        train_size = len(X_train)
        valid_size = len(X_valid)
        
        self.model.fit(X_train.reshape(train_size, self.image_base_size*self.image_base_size*self.channels), y_train)
        y_pred = self.model.predict(X_valid.reshape(valid_size, self.image_base_size*self.image_base_size*self.channels))
        print('Validation f2: %.2f' % fbeta_score(y_pred, y_valid, beta=2, average='samples'))


    def predict(self, X_test):
        '''
        Predicts labels using a fitted model
        '''
        
        test_size = len(X_test)
        return self.model.predict(X_test.reshape(test_size, self.image_base_size*self.image_base_size*self.channels))
