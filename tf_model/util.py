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


def create_jpg_data(image_base_size=28):
    """
    This function assumes that you have the folders:

    ./Data/train-jpg
    ./Data/test-jpg

    and the file:

    ./Data/train.csv

    it uses the param image_base_size (max = 256) to resize the images in
    both folders and transform them into a np array.
    The labels of the file ./Data/train.csv are also transformed
    into an array. At the end it saves a pickle file into
    the folder Data

    :type image_base_size: int
    """
    BASE_DIR = './Data'
    train_dir = os.path.join(BASE_DIR, 'train-jpg')
    test_dir = os.path.join(BASE_DIR, 'test-jpg')
    train_img_names = sorted(glob(train_dir + '/*.jpg'))
    test_img_names = sorted(glob(test_dir + '/*.jpg'))
    label_file_name = os.path.join(BASE_DIR, 'train.csv')
    label_file = pd.read_csv(label_file_name)
    label_file = label_file.sort_values('image_name')
    vec = CountVectorizer(min_df=1)
    labels = vec.fit_transform(label_file['tags'])

    X_train_names = [os.path.basename(img_name).replace('.jpg', '') for img_name in train_img_names]
    X_test_names = [os.path.basename(img_name).replace('.jpg', '') for img_name in test_img_names]

    X_train_names = sorted(X_train_names)
    X_test_names = sorted(X_test_names)

    assert label_file['image_name'].tolist() == X_train_names
    pd.DataFrame(train_img_names, columns=['image_name']).to_csv(os.path.join(BASE_DIR, 'train_img_names.csv'))
    pd.DataFrame(test_img_names, columns=['image_name']).to_csv(os.path.join(BASE_DIR, 'test_img_names.csv'))
    print('X_train: load')
    X_train = []
    train_size = len(train_img_names)
    for i, img_name in enumerate(train_img_names):
        sys.stdout.write('\r{} of {}'.format(i+1, train_size))
        sys.stdout.flush()
        image = np.array(Image.open(img_name).resize((image_base_size, image_base_size)))[:, :, :3]
        image = image_standardization(image)
        X_train.append(image)
    X_train = np.array(X_train)
    print('\ny_train: load')
    y_train = labels.toarray()
    print('X_test: load')
    X_test = []
    test_size = len(test_img_names)
    for i, img_name in enumerate(test_img_names):
        sys.stdout.write('\r{} of {}'.format(i+1, test_size))
        sys.stdout.flush()
        image = np.array(Image.open(img_name).resize((image_base_size, image_base_size)))[:, :, :3]
        image = image_standardization(image)
        X_test.append(image)
    X_test = np.array(X_test)
    file_name = "data_jpg_" + str(image_base_size) + ".pickle"
    file_name = os.path.join(BASE_DIR, file_name)
    file = open(file_name, 'wb')
    dict_file = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test}
    pickle.dump(dict_file, file)
    file.close()
