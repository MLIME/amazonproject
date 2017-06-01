#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def args(self):
        '''
        Lists model parameters
        '''
        pass

    @abstractmethod
    def initialize(self, args):
        '''
        Initialize model parameters
        '''
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a model
        '''

        pass

    @abstractmethod
    def predict(self, X_test):
        '''
        Predicts labels using a fitted model
        '''
        pass
