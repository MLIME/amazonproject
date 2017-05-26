import unittest
import os
import sys
import inspect
import numpy as np
import subprocess


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
toolsdir = os.path.join(parentdir, "tools")
sys.path.insert(0, toolsdir)

from DataManager import DataManager
from util import run_test


class TestDataManager(unittest.TestCase):
    """
    Class that test the DataManager class
    """
    @classmethod
    def setUpClass(cls):
        cls.test_pred = np.array([[0, 1, 0, 1, 1, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 1, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1]])

        cls.test_add_pred = np.array([[1, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0, 0, 1],
                                      [1, 1, 0, 1, 0, 0, 0],
                                      [0, 1, 0, 0, 1, 1, 1]])
        cls.datadir = os.path.join(parentdir, "toy_data")
        file_path = os.path.join(cls.datadir, "toy_submission.csv")

        cls.bashCommand = "diff submission.csv " + file_path
        cls.path_folder = os.path.join(parentdir, "toy_data")

    def test_submission(self):
        """
        Test of the submission function
        """
        my_data = DataManager(TestDataManager.datadir,
                              label_file_name="toy_train.csv")
        test = True
        my_data.get_submission(TestDataManager.test_pred,
                               TestDataManager.test_add_pred)
        try:
            output = subprocess.check_output(['bash',
                                              '-c',
                                              TestDataManager.bashCommand])
            output = str(output, 'utf-8')
            test = output == ""
        except subprocess.CalledProcessError:
            test = False
        try:
            os.remove('submission.csv')
        except OSError:
            pass
        self.assertTrue(test)


if __name__ == "__main__":
    run_test(TestDataManager,
             "\n=== Running test for the TestDataManager class ===\n")
