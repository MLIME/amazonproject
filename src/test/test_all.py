import os
import sys
import inspect
from TestDataManager import TestDataManager
from TestCNN import TestCNN

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
libdir = os.path.join(parentdir, "lib")
sys.path.insert(0, libdir)

from util import run_test


def main():
    run_test(TestDataManager,
             "\n=== Running test for the TestDataManager class ===\n")
    run_test(TestCNN,
             "\n=== Running test for the CNN model ===\n")

if __name__ == "__main__":
    main()
