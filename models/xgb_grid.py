#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from base_model import BaseModel
from utils import get_timestamp
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

class XGBGridModel(BaseModel):
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

        param_grid = [{'estimator__n_estimators': [100, 500], 'estimator__max_depth': [3, 7], 'estimator__learning_rate': [0.001, 0.1]}]
        f2_scorer = make_scorer(fbeta_score, beta=2, average='samples')

        xgb = XGBClassifier(silent=False)
        onevrest = OneVsRestClassifier(xgb)
        self.model = GridSearchCV(onevrest, param_grid, cv=2, scoring=f2_scorer, n_jobs=16)


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a model
        '''
        
        train_size = len(X_train)
        valid_size = len(X_valid)
        
        self.model.fit(X_train.reshape(train_size, self.image_base_size*self.image_base_size*self.channels), y_train)

        print(self.model.best_params_)
        print(self.model.best_score_)

        y_pred = self.model.best_estimator_.predict(X_valid.reshape(valid_size, self.image_base_size*self.image_base_size*self.channels))

        print('Validation f2: %.2f' % fbeta_score(y_pred, y_valid, beta=2, average='samples'))


    def predict(self, X_test):
        '''
        Predicts labels using a fitted model
        '''
        
        test_size = len(X_test)
        return self.model.best_estimator_.predict(X_test.reshape(test_size, self.image_base_size*self.image_base_size*self.channels))
