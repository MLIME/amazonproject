import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
import pickle
import sys
from PIL import Image
from util import image_standardization


class JpgTransformer:
    """
    Class to transform all the jpg data and
    to write the submission file

    :type image_base_size: int
    :type base_dir: str
    """
    def __init__(self,
                 image_base_size=28,
                 base_dir='./Data'):
        self.image_base_size = image_base_size
        self.base_dir = base_dir
        file_name = "data_jpg_" + str(image_base_size) + ".pickle"
        self.file_name = os.path.join(base_dir, file_name)
        if not os.path.exists(self.file_name):
            self.create_jpg_data()

    def create_jpg_data(self):
        """
        This method assumes that you have the folders:

        <base_dir>/train-jpg
        <base_dir>/test-jpg

        and the file:

        <base_dir>/train.csv

        it uses the param self.image_base_size (max = 256) to resize the
        images in both folders and transform them into a np array.
        The labels of the file <base_dir>/train.csv are also transformed
        into an array. At the end it saves a pickle file into
        the folder Data
        """
        base_dir = self.base_dir
        image_base_size = self.image_base_size
        train_dir = os.path.join(base_dir, 'train-jpg')
        test_dir = os.path.join(base_dir, 'test-jpg')
        train_img_names = sorted(glob(train_dir + '/*.jpg'))
        test_img_names = sorted(glob(test_dir + '/*.jpg'))
        label_file_name = os.path.join(base_dir, 'train.csv')
        label_file = pd.read_csv(label_file_name)
        label_file = label_file.sort_values('image_name')
        vec = CountVectorizer(min_df=1)
        labels = vec.fit_transform(label_file['tags'])

        X_train_names = [os.path.basename(img_name).replace('.jpg', '') for img_name in train_img_names]
        X_test_names = [os.path.basename(img_name).replace('.jpg', '') for img_name in test_img_names]

        X_train_names = sorted(X_train_names)
        X_test_names = sorted(X_test_names)

        assert label_file['image_name'].tolist() == X_train_names
        pd.DataFrame(train_img_names, columns=['image_name']).to_csv(os.path.join(base_dir, 'train_img_names.csv'))
        pd.DataFrame(test_img_names, columns=['image_name']).to_csv(os.path.join(base_dir, 'test_img_names.csv'))
        print('X_train: load')
        X_train = []
        train_size = len(train_img_names)
        for i, img_name in enumerate(train_img_names):
            sys.stdout.write('\r{} of {}'.format(i + 1, train_size))
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
            sys.stdout.write('\r{} of {}'.format(i + 1, test_size))
            sys.stdout.flush()
            image = np.array(Image.open(img_name).resize((image_base_size, image_base_size)))[:, :, :3]
            image = image_standardization(image)
            X_test.append(image)
        X_test = np.array(X_test)
        file = open(self.file_name, 'wb')
        dict_file = {'X_train': X_train,
                     'y_train': y_train,
                     'X_test': X_test,
                     'vec': vec,
                     'test_img_names': test_img_names}
        pickle.dump(dict_file, file)
        file.close()

    def create_submission(self, predictions):
        with open(self.file_name, 'rb') as s:
            d = pickle.load(s)
            pass
        X_test = d['X_test']
        vec = d['vec']
        test_img_names = d['test_img_names']
        del d
        test_size = len(X_test)
        assert test_size == len(predictions)
        pred_labels = [list(vec.inverse_transform(i)[0]) for i in predictions]
        submission_file = open('submission.csv', 'w')
        submission_file.write('id,image_name,tags\n')

        for i in range(test_size):
            id_file = test_img_names[i].split('_')[1][:-4]
            labels_s = ' '.join(pred_labels[i])
            s = id_file + ',' + "test_" + id_file + ',' + labels_s
            sys.stdout.write('\r{} of {}'.format(i + 1, test_size))
            sys.stdout.flush()
            submission_file.write(s)
            submission_file.write('\n')

        submission_file.close()

        submission_data = pd.read_csv('submission.csv')
        submission_data = submission_data.sort_values('id').drop('id', 1)
        submission_data.to_csv('submission.csv', index=False)
