#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K


class KerasMetrics:
    def precision(self, y_true, y_pred):
        '''
        calculates precision

        :type y_true: np array
        :type y_pred: np array
        :rtype: float
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(self, y_true, y_pred):
        '''
        calculates recall

        :type y_true: np array
        :type y_pred: np array
        :rtype: float
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def fbeta_score(self, y_true, y_pred, beta=1):
        '''
        calculates f beta score

        :type y_true: np array
        :type y_pred: np array
        :type beta: int
        :rtype: float
        '''
        if beta < 0:
            info = 'The lowest choosable beta is zero (only precision).'
            raise ValueError(info)

        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

    def f2_score(self, y_true, y_pred):
        '''
        calculates f 2 score

        :type y_true: np array
        :type y_pred: np array
        :rtype: float
        '''
        return self.fbeta_score(y_true, y_pred, beta=2)
