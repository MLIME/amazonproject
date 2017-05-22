import pandas as pd
import pickle
from Config import Config
from CNN import CNNModel, train_model, check_valid, test_prediction
from DataHolder import DataHolder
from Config import Config
from JpgTransformer import JpgTransformer
from util import randomize_in_place
from collections import Counter

transformer = JpgTransformer(image_base_size=28)
file_name = transformer.file_name

with open(file_name, 'rb') as s:
    d = pickle.load(s)
    pass
X_train = d['X_train']
y_train = d['y_train']
X_test = d['X_test']
del d

randomize_in_place(X_train, y_train, 0)
X_valid, y_valid = X_train[40000: 40479], y_train[40000: 40479]
X_train, y_train = X_train[0:40000], y_train[0:40000]
print("image shape train = ", X_train.shape)
print("label shape train = ", y_train.shape)
print("image shape valid = ", X_valid.shape)
print("label shape valid = ", y_valid.shape)

lr = 0.0928467676
my_dataholder = DataHolder(X_train,
                           y_train,
                           X_valid,
                           y_valid,
                           X_test)
my_config = Config(batch_size=120,
                   learning_rate=lr,
                   image_size=28)
my_model = CNNModel(my_config, my_dataholder)
train_model(my_model, my_dataholder, 10001, 1000)
print("check_valid = ", check_valid)

test_pred = test_prediction(my_model)
transformer.create_submission(test_pred)

submission_data = pd.read_csv('submission.csv')
c = dict(Counter(list(submission_data["tags"])))
print("Showing the frequency of the predictions:\n")
print(c)
