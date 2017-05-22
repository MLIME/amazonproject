import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
import os
import unittest
import time
import matplotlib.pyplot as plt
import sys
from PIL import Image

timing = {}


def get_time(f, args=[]):
    """
    After using timeit we can get the duration of the function f
    when it was applied in parameters args. Normally it is expected
    that args is a list of parameters, but it can be also a single parameter.

    :type f: function
    :type args: list
    :rtype: float
    """
    if type(args) != list:
        args = [args]
    key = f.__name__
    if args != []:
        key += "-" + "-".join([str(arg) for arg in args])
    return timing[key]


def timeit(index_args=[]):

    def dec(method):
        """
        Decorator for time information
        """

        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            timed.__name__ = method.__name__
            te = time.time()
            fkey = method.__name__
            for i, arg in enumerate(args):
                if i in index_args:
                    fkey += "-" + str(arg)
            timing[fkey] = te - ts
            return result
        return timed
    return dec


def F_beta(prediction, truth, beta=2):
    """
    F-score function for any positive real beta, reference:
    https://en.wikipedia.org/wiki/F1_score

    :type prediction: np array
    :type truth: np array
    :type beta: np array
    :rtype: float
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] == truth[i] == 1:
            tp += 1
        if prediction[i] > truth[i]:
            fp += 1
        if prediction[i] < truth[i]:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    numerator = (1 + beta ** 2) * (precision * recall)
    denominator = ((beta ** 2) * (precision)) + recall
    return numerator / denominator


def plot9images(images, cls_true, img_shape, cls_pred=None):
    """
    Function to show 9 images with their respective classes.
    If cls_pred is an array, you can see the image and the prediction.

    :type images: np array
    :type cls_true: np array
    :type img_shape: np array
    :type cls_prediction: None or np array
    """
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def randomize_in_place(list1, list2, init):
    """
    Function to randomize two lists the same way.
    Usualy this functions is used when list1 = dataset,
    and list2 = labels.

    :type list1: list
    :type list2: list
    :type init: int
    """
    np.random.seed(seed=init)
    np.random.shuffle(list1)
    np.random.seed(seed=init)
    np.random.shuffle(list2)


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.

    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def get_log_path():
    """
    Function to create one unique path for each model.
    This path is created by using the specific time that
    the function is called.

    :rtype: str
    """
    log_basedir = 'logs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')
    log_path = os.path.join(log_basedir, run_label)
    return log_path


def image_standardization(image):
    """
    Transform an image of tipe int [0,255]
    into an array of floats

    :type image: np array
    :rtype: np array
    """
    image = image.astype('float32')
    return image / 255


def get_batch(data, labels, batch_size):
    """
    Given one dataset data and an array of labels label,
    this function returns a batch of size batch_size
    :type data: np array
    :type labels: np array
    :type batch_size: int
    :rtype: tuple of np arrays
    """
    random_indices = np.random.randint(data.shape[0], size=batch_size)
    return data[random_indices], labels[random_indices]
