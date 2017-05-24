from base_model import BaseModel

class SampleModel(BaseModel):
    def __init__(self):
        self.args_names = ['--arg1', '--arg2']
        self.args_desc = ['arg1 desc', 'arg2 desc']
    
    def args(self):
        '''
        Lists model parameters
        '''
        return [(k, v) for (k, v) in zip(self.args_names, self.args_desc)]


    def initialize(self, args):
        '''
        Initialize model parameters
        '''
        pass


    def fit(self, X_train, y_train, X_valid, y_valid):
        '''
        Fits a model
        '''
        pass


    def predict(self, X_test):
        '''
        Predicts labels using a fitted model
        '''
        pass