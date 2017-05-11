#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from PIL import Image
from skimage.io import ImageCollection
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob

BASE_DIR = './Data'

image_base_size = 28

train_dir = os.path.join(BASE_DIR, 'train-jpg')
test_dir = os.path.join(BASE_DIR, 'test-jpg')

train_img_names = sorted(glob(train_dir + '/*.jpg'))
test_img_names = sorted(glob(test_dir + '/*.jpg'))

#load train labels
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

#convert data to numpy

pd.DataFrame(train_img_names, columns=['image_name']).to_csv(os.path.join(BASE_DIR, 'train_img_names.csv'))
pd.DataFrame(test_img_names, columns=['image_name']).to_csv(os.path.join(BASE_DIR, 'test_img_names.csv'))

print('X_train: load')
X_train = np.array([np.array(Image.open(img_name).resize((image_base_size, image_base_size))) for img_name in train_img_names])

print('X_train: save')
np.save(BASE_DIR + '/X_train.npy', X_train)
X_train = None

print('y_train: load')
y_train = labels.toarray()

print('y_train: save')
np.save(BASE_DIR + '/y_train.npy', y_train)

print('X_test: load')
X_test = np.array([np.array(Image.open(img_name).resize((image_base_size, image_base_size))) for img_name in test_img_names])

print('X_test: save')
np.save(BASE_DIR + '/X_test.npy', X_test)
X_test = None

