import os
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import sys
try:
    from util import image_standardization, get_timestamp
except ImportError:
    from tools.util import image_standardization, get_timestamp


class DataManager:
    """
    sansklansklas
    """
    def __init__(self,
                 base_dir,
                 image_base_size=28,
                 channels=3,
                 file_ext='jpg',
                 label_file_name='train_v2.csv',
                 max_image_value=255):
        assert file_ext in ['jpg', 'tif']
        self.base_dir = base_dir
        self.image_base_size = image_base_size
        self.channels = channels
        self.file_ext = file_ext
        self.label_file_name = label_file_name
        self.max_image_value = max_image_value
        self.timestamp = get_timestamp()

        self.train_dir = os.path.join(base_dir, 'train-jpg')
        self.test_dir = os.path.join(base_dir, 'test-jpg')
        self.test_dir_add = os.path.join(base_dir, 'test-jpg-additional')
        pickle_name = 'data' + str(image_base_size) + file_ext + '.pickle'
        self.pickle_file_name = os.path.join(self.base_dir, pickle_name)
        self.label_file_name = os.path.join(base_dir, self.label_file_name)
        self.vec = CountVectorizer(min_df=1)
        self.create_data()

    def load_labels(self):
        """
        Transform the labels of the file self.label_file_name into a array
        of 1s and 0s.
        """
        self.label_file = pd.read_csv(self.label_file_name)
        self.label_file = self.label_file.sort_values('image_name')
        self.labels = self.vec.fit_transform(self.label_file['tags'])
        self.labels = np.clip(self.labels.toarray(), 0, 1)
        self.num_categories = self.labels.shape[1]
        self.label_names = np.array(self.vec.get_feature_names())

    def load_file_list(self):
        """
        Sort the name of all the files
        """
        self.train_img_names = sorted(glob(self.train_dir + '/*.' + self.file_ext))
        self.test_img_names = sorted(glob(self.test_dir + '/*.' + self.file_ext))
        self.test_img_names_add = sorted(glob(self.test_dir_add + '/*.' + self.file_ext))

        self.X_train_names = [os.path.basename(img_name).replace('.jpg', '') for img_name in self.train_img_names]
        self.X_test_names = [os.path.basename(img_name).replace('.jpg', '') for img_name in self.test_img_names]
        self.X_test_names_add = [os.path.basename(img_name).replace('.jpg', '') for img_name in self.test_img_names_add]

        self.X_train_names = sorted(self.X_train_names)
        self.X_test_names = sorted(self.X_test_names)
        self.X_test_names_add = sorted(self.X_test_names_add)

        assert self.label_file['image_name'].tolist() == self.X_train_names

    def read_image(self, img_name):
        """
        read image, resize it, and normalize it
        """
        shape = (self.image_base_size, self.image_base_size)
        image = np.array(Image.open(img_name).resize(shape))[:, :, : self.channels]
        return image

    def load_images(self):
        """
        load all images
        """
        if not os.path.exists(self.pickle_file_name):
            X_train = np.array([self.read_image(img_name) for img_name in self.train_img_names])
            X_train = image_standardization(X_train, self.max_image_value)
            X_test = np.array([self.read_image(img_name) for img_name in self.test_img_names])
            X_test = image_standardization(X_test, self.max_image_value)
            X_test_add = np.array([self.read_image(img_name) for img_name in self.test_img_names_add])
            X_test_add = image_standardization(X_test_add, self.max_image_value)

            self.data = dict()
            self.data['y_train'] = self.labels
            self.data['X_train'] = X_train
            self.data['X_test'] = X_test
            self.data['X_test_add'] = X_test_add

            with open(self.pickle_file_name, 'wb') as pickle_file:
                pickle.dump(self.data, pickle_file)
        else:
            with open(self.pickle_file_name, 'rb') as pickle_file:
                self.data = pickle.load(pickle_file)

    def create_data(self):
        """
        This method assumes that you have the folders:

        <base_dir>/train-jpg
        <base_dir>/test-jpg
        <base_dir>/test-jpg-additional

        and the file:

        <base_dir>/train.csv

        it uses the param self.image_base_size (max = 256) to resize the
        images in both folders and transform them into a np array.
        The labels of the file <base_dir>/train.csv are also transformed
        into an array. At the end it saves a pickle file into
        the folder Data
        """
        self.load_labels()
        self.load_file_list()
        self.load_images()

    def save_submission_file(self, y_pred, add=False, verbose=False):
            """
            save submission
            """
            if add:
                image_names = self.X_test_names_add
                file_name = 'submission2.csv'
                car = "file_"
            else:
                image_names = self.X_test_names
                file_name = 'submission1.csv'
                car = "test_"
            test_size = len(image_names)
            assert test_size == len(y_pred)
            pred_labels = [list(self.vec.inverse_transform(i)[0]) for i in y_pred]
            submission_file = open(file_name, 'w')
            submission_file.write('id,image_name,tags\n')

            for i in range(test_size):
                id_file = image_names[i].split('_')[1]
                labels_s = ' '.join(pred_labels[i])
                s = id_file + ',' + car + id_file + ',' + labels_s
                if verbose:
                    sys.stdout.write('\r{} of {}'.format(i + 1, test_size))
                    sys.stdout.flush()
                submission_file.write(s)
                submission_file.write('\n')

            submission_file.close()

            submission_data = pd.read_csv(file_name)
            submission_data = submission_data.sort_values('id').drop('id', 1)
            submission_data.to_csv(file_name, index=False)

    def get_submission(self, y_pred, y_pred_add):
        """
        Combine submissions
        """
        self.save_submission_file(y_pred, add=False)
        self.save_submission_file(y_pred_add, add=True)
        df1 = pd.read_csv('submission1.csv')
        df2 = pd.read_csv('submission2.csv')
        frames = [df1, df2]
        df = pd.concat(frames)
        df.fillna("clear primary", inplace=True)
        df.to_csv("submission.csv", index=False)
        try:
            os.remove('submission1.csv')
        except OSError:
            pass
        try:
            os.remove('submission2.csv')
        except OSError:
            pass
