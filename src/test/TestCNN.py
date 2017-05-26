import unittest
import os
import sys
import inspect
import shutil


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
toolsdir = os.path.join(parentdir, "tools")
sys.path.insert(0, toolsdir)
sys.path.insert(0, parentdir)

from DataManager import DataManager
from util import run_test, randomize_in_place
from DataHolder import DataHolder
from Config import Config
from CNN import CNNModel, train_model, check_valid, test_prediction


class TestCNN(unittest.TestCase):
    """
    Class that test the DataManager class
    """
    @classmethod
    def setUpClass(cls):
        cls.datadir = os.path.join(parentdir, "toy_data")
        cls.my_data = DataManager(cls.datadir,
                                  label_file_name="toy_train.csv")

    def test_train_to_submission(self):
        """
        Testing the whole process from training to submission
        """
        X_train = TestCNN.my_data.data['X_train']
        y_train = TestCNN.my_data.data['y_train']
        X_test = TestCNN.my_data.data['X_test']
        X_test_add = TestCNN.my_data.data['X_test_add']
        randomize_in_place(X_train, y_train, 0)
        X_valid, y_valid = X_train, y_train
        X_train, y_train = X_train, y_train
        lr = 0.0928467676
        my_dataholder = DataHolder(X_train,
                                   y_train,
                                   X_valid,
                                   y_valid,
                                   X_test,
                                   X_test_add)

        my_config = Config(batch_size=1,
                           learning_rate=lr,
                           image_size=28,
                           num_labels=7)
        my_model = CNNModel(my_config, my_dataholder)
        train_model(my_model, my_dataholder, 101, 50, verbose=False)
        test_pred = test_prediction(my_model)
        test_pred_add = test_prediction(my_model, add=True)
        TestCNN.my_data.get_submission(test_pred, test_pred_add)
        self.assertTrue(check_valid(my_model) > 0.1)
        self.assertTrue(os.path.exists("submission.csv"))
        os.remove("submission.csv")
        shutil.rmtree("checkpoints/")
        shutil.rmtree("logs/")


if __name__ == "__main__":
    run_test(TestCNN,
             "\n=== Running test for the CNN model ===\n")
