#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from glob import glob
from skimage import io, transform, img_as_ubyte, filters, color
from skimage.feature import greycomatrix, local_binary_pattern
from scipy import misc
from models.utils import get_timestamp
import spectral
import warnings

warnings.filterwarnings('ignore')

class DataManager:
    def __init__(self, base_dir, model_name, train_dir_name, test_dir_name,
        file_ext, image_base_size, channels, bit_depth, label_file_name, 
        apply_sobel=False, apply_fft=False, apply_lbp=False, apply_co=False, apply_kmeans=False):
        
        assert file_ext in ['jpg', 'tif']

        spectral.settings.show_progress = False

        self.base_dir = base_dir
        self.model_name = model_name
        self.file_ext = file_ext
        self.image_base_size = image_base_size
        
        self.channels = channels
        self.bit_depth = bit_depth
        self.max_image_value = 255
        
        self.timestamp = get_timestamp()
        
        self.pickle_file_name = os.path.join(self.base_dir, 'data_' + file_ext + '.pickle')
        
        self.submission_file_name = self.model_name + '_submission.csv'
        
        self.train_dir = os.path.join(base_dir, train_dir_name)
        self.test_dir = os.path.join(base_dir, test_dir_name)
        
        self.label_file_name = os.path.join(base_dir, label_file_name)
        self.vec = CountVectorizer(min_df=1)
        
        self.apply_sobel = apply_sobel
        self.apply_fft = apply_fft
        self.apply_lbp = apply_lbp
        self.apply_co = apply_co
        self.apply_kmeans = apply_kmeans

        file_tag = '_'
        filters = 0

        if self.apply_sobel:
            file_tag += 'sobel_'
            filters += 1

        if self.apply_fft:
            file_tag += 'fft_'
            filters += 1

        if self.apply_lbp:
            file_tag += 'lbp_'
            filters += 1

        if self.apply_co:
            file_tag += 'co_'
            filters += 1

        if self.apply_kmeans:
            file_tag += 'kmeans_'
            filters += 1

        if filters > 3:
            raise Exception('Maximum number of filters is 3')
        else:
            self.output_channels = min(self.channels + filters, 4)


        self.X_train_mmap_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.output_channels) + '_' + str(self.image_base_size) + file_tag + 'X_train.mmap')
        self.X_test_mmap_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.output_channels) + '_' + str(self.image_base_size) + file_tag + 'X_test.mmap')
        self.X_valid_mmap_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.output_channels) + '_' + str(self.image_base_size) + file_tag + 'X_valid.mmap')
    
        self.y_train_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.output_channels) + '_' + str(self.image_base_size) + file_tag + 'y_train.npy')
        self.y_valid_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.output_channels) + '_' + str(self.image_base_size) + file_tag + 'y_valid.npy')

    
    def load_labels(self):
        self.label_file = pd.read_csv(self.label_file_name)
        self.label_file = self.label_file.sort_values('image_name')
        self.labels = self.vec.fit_transform(self.label_file['tags'])
        self.labels = np.clip(self.labels.toarray(), 0, 1)

        self.num_categories = self.labels.shape[1]
        
        self.label_names = np.array(self.vec.get_feature_names())

    
    def load_file_list(self):
        self.train_img_names = []
        
        for r in self.label_file.iterrows():
            img_name = r[1][0]
            self.train_img_names.append(os.path.join(self.train_dir, img_name + '.' + self.file_ext))

        self.test_img_names = sorted(glob(self.test_dir + '/*.' + self.file_ext))
        
        self.X_train_names = [os.path.basename(img_name).replace('.' + self.file_ext, '') for img_name in self.train_img_names]
        self.X_test_names = [os.path.basename(img_name).replace('.' + self.file_ext, '') for img_name in self.test_img_names]
        
        self.X_test_names = sorted(self.X_test_names)
        
        assert self.label_file['image_name'].tolist() == self.X_train_names

    
    def read_image(self, img_name):
        if self.file_ext == 'tif' and self.bit_depth == 16:
            img = transform.resize(img_as_ubyte(io.imread(img_name)), (self.image_base_size, self.image_base_size, self.channels), preserve_range=True)
        else:
            img = transform.resize(io.imread(img_name), (self.image_base_size, self.image_base_size, self.channels), preserve_range=True)

        new_img = np.zeros((self.image_base_size, self.image_base_size, self.output_channels)).astype('uint8')
        
        filter_channels = []
        
        if self.apply_sobel:
            filter_channels.append(self.apply_sobel_filter(img))

        if self.apply_fft:
            filter_channels.append(self.apply_fft_filter(img))

        if self.apply_lbp:
            filter_channels.append(self.apply_lbp_filter(img))

        if self.apply_co:
            filter_channels.append(self.apply_co_filter(img))

        if self.apply_kmeans:
            filter_channels.append(self.apply_kmeans_filter(img))

        filter_channels.append(img[:,:,1])
        filter_channels.append(img[:,:,2])
        filter_channels.append(img[:,:,0])

        for i in range(self.output_channels):
            new_img[:,:,i] = filter_channels[i]

        return np.array(new_img, dtype='float32')


    def apply_kmeans_filter(self, img):
        (m, c) = spectral.kmeans(img, 5, 20)
        return m


    def apply_sobel_filter(self, img):
        img_gray = color.rgb2gray(img)
        img_gray = np.floor(img_gray * self.max_image_value).astype('uint8')

        img_sobel = img_as_ubyte(filters.sobel(img[:,:,1] / self.max_image_value))

        return img_sobel


    def apply_co_filter(self, img):
        img_gray = color.rgb2gray(img)
        img_gray = np.floor(img_gray * self.max_image_value).astype('uint8')

        img_glcm = greycomatrix(img_gray, [128], [0], self.max_image_value+1)
        img_glcm = transform.resize(img_as_ubyte(img_glcm[:,:,0,0] / img_glcm[:,:,0,0].max()), (self.image_base_size, self.image_base_size), preserve_range=True)

        return img_glcm


    def apply_fft_filter(self, img):
        img_gray = color.rgb2gray(img)
        img_gray = np.floor(img_gray * self.max_image_value).astype('uint8')

        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        img_fft = 20 * np.log(np.abs(fshift))
        img_fft = np.clip(img_fft, 0, self.max_image_value)/img_fft.max()*255

        return img_fft
   

    def apply_lbp_filter(self, img):
        img_lbp = img_as_ubyte(local_binary_pattern(img[:,:,1], 64, 8, method="uniform") / self.max_image_value)

        return img_lbp

    
    def load_images_mmap(self, validation_split=0.2):
        assert validation_split > 0.0
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
        train_imgs, valid_imgs = list(sss.split(self.labels, self.labels))[0]
        
        num_test_imgs  = len(self.test_img_names)
        num_train_imgs = len(train_imgs)
        num_valid_imgs = len(valid_imgs)
        
        print('Train: %d, Valid: %d, Test: %d' % (num_train_imgs, num_valid_imgs, num_test_imgs))
        
        if not os.path.exists(self.X_train_mmap_file):
            self.files_to_mmap(list(np.array(self.train_img_names)[train_imgs]), self.X_train_mmap_file)
        
        if not os.path.exists(self.X_valid_mmap_file):
            self.files_to_mmap(list(np.array(self.train_img_names)[valid_imgs]), self.X_valid_mmap_file)           

        if not os.path.exists(self.X_test_mmap_file):
            self.files_to_mmap(self.test_img_names, self.X_test_mmap_file)
        
        if not os.path.exists(self.y_train_file):
            y_train = self.labels[train_imgs]
            np.save(self.y_train_file, y_train)

        if not os.path.exists(self.y_valid_file):           
            y_valid = self.labels[valid_imgs]
            np.save(self.y_valid_file, y_valid)
        
        X_train = self.load_mmap_file(self.X_train_mmap_file, num_train_imgs)
        y_train = np.load(self.y_train_file)
        
        X_valid = self.load_mmap_file(self.X_valid_mmap_file, num_valid_imgs)
        y_valid = np.load(self.y_valid_file)
        
        X_test = self.load_mmap_file(self.X_test_mmap_file, num_test_imgs)
        
        self.data = dict()
        self.data['X_train'] = X_train
        self.data['y_train'] = y_train
        self.data['X_valid'] = X_valid
        self.data['y_valid'] = y_valid
        self.data['X_test'] = X_test

    
    def load_mmap_file(self, file_name, length):
        return np.memmap(file_name, dtype='float32', mode = 'r', shape=(length, self.image_base_size, self.image_base_size, self.output_channels))
    
   
    def files_to_mmap(self, img_names, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)
        
        print('Writing %d images to %s (%d x %d x %d)' % (len(img_names), file_name, self.image_base_size, self.image_base_size, self.output_channels))
        
        os.mknod(file_name)
        data = np.memmap(file_name, dtype='float32', mode = 'r+', shape=(len(img_names), self.image_base_size, self.image_base_size, self.output_channels))
        
        with click.progressbar(range(len(img_names))) as vals:
            for i in vals:
                data[i] = np.array(self.read_image(img_names[i]) / self.max_image_value, dtype='float32')
                
                if i % 1000 == 0 and i > 0:
                    data.flush()
        
        data.flush()
        data._mmap.close()
    
    
    def get_labels(self, y):
#        y[y >= 0.5] = 1
#        y[y < 0.5] = 0
        thresholds = [0.9, 0.8, 0.9, 0.8, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]
        y = (y > thresholds).astype('uint8')
        return np.array(self.vec.inverse_transform(y))

    
    def save_submission_file(self, y_pred):
        pred_labels = self.get_labels(y_pred)
        sub_file_name = os.path.join(self.base_dir, self.timestamp + '_' + self.submission_file_name)
        
        with open(sub_file_name, 'w') as submission_file:
            submission_file.write('id,image_name,tags\n')
            
            for i in range(len(pred_labels)):
                file_name = os.path.basename(self.test_img_names[i]).replace('.' + self.file_ext, '')
                id_file = file_name.split('_')[1]
                s = id_file + ',' + file_name + ',' + ' '.join(pred_labels[i])
                submission_file.write(s)
                submission_file.write('\n')
        
        submission_data = pd.read_csv(sub_file_name)
        submission_data = submission_data.sort_values('id').drop('id', 1)
        submission_data.to_csv(sub_file_name, index=False)


    def save_preds_to_matrix(self, y_pred, file_name):
        pred_file_name = os.path.join(self.base_dir, self.timestamp + '_' + file_name + '.npy')
        np.save(pred_file_name, y_pred)

