from Config import Config
from CNN import CNNModel, train_model, check_valid, test_prediction
from DataHolder import DataHolder
from tools.util import randomize_in_place
from tools.DataManager import DataManager


my_data = DataManager("data/", image_base_size=55)
X_train = my_data.data['X_train']
y_train = my_data.data['y_train']
X_test = my_data.data['X_test']
X_test_add = my_data.data['X_test_add']
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
                           X_test,
                           X_test_add)
my_config = Config(batch_size=120,
                   learning_rate=lr,
                   image_size=55)
my_model = CNNModel(my_config, my_dataholder)
train_model(my_model, my_dataholder, 10001, 1000)
print(check_valid(my_model))
test_pred = test_prediction(my_model)
test_pred_add = test_prediction(my_model, add=True)
my_data.get_submission(test_pred, test_pred_add)
