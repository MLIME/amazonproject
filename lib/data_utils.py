#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
import cv2
from scipy import misc
from models.utils import get_timestamp


class DataManager:
	def __init__(self, base_dir, model_name, train_dir_name, test_dir_name, file_ext, image_base_size, channels, bit_depth, label_file_name):
		assert file_ext in ['jpg', 'tif']

		self.base_dir = base_dir
		self.model_name = model_name
		self.file_ext = file_ext
		self.image_base_size = image_base_size
		self.channels = channels
		self.bit_depth = bit_depth
		self.max_image_value = (2 ** bit_depth) - 1
		
		self.pickle_file_name = os.path.join(self.base_dir, 'data_' + file_ext + '.pickle')
		
		self.submission_file_name = self.model_name + '_submission.csv'
		
		self.train_dir = os.path.join(base_dir, train_dir_name)
		self.test_dir = os.path.join(base_dir, test_dir_name)
		
		self.label_file_name = os.path.join(base_dir, label_file_name)
		self.vec = CountVectorizer(min_df=1)
		
		self.X_train_mmap_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.channels) + '_' + str(self.image_base_size) + '_X_train.mmap')
		self.X_test_mmap_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.channels) + '_' + str(self.image_base_size) + '_X_test.mmap')
		self.X_valid_mmap_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.channels) + '_' + str(self.image_base_size) + '_X_valid.mmap')
		
		self.y_train_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.channels) + '_' + str(self.image_base_size) + '_y_train.npy')
		self.y_valid_file = os.path.join(self.base_dir, self.file_ext + '_' + str(self.channels) + '_' + str(self.image_base_size) + '_y_valid.npy')

	
	def load_labels(self):
		self.label_file = pd.read_csv(self.label_file_name)
		self.label_file = self.label_file.sort_values('image_name')
		self.labels = self.vec.fit_transform(self.label_file['tags'])
        self.labels = np.clip(self.labels.toarray(), 0, 1)
        
        self.label_names = np.array(vec.get_feature_names())
        
		self.num_categories = self.labels.toarray().shape[1]

	
	def load_file_list(self):
		self.train_img_names = sorted(glob(self.train_dir + '/*.' + self.file_ext))
		self.test_img_names = sorted(glob(self.test_dir + '/*.' + self.file_ext))
		
		self.X_train_names = [os.path.basename(img_name).replace('.' + self.file_ext, '') for img_name in self.train_img_names]
		self.X_test_names = [os.path.basename(img_name).replace('.' + self.file_ext, '') for img_name in self.test_img_names]
		
		self.X_train_names = sorted(self.X_train_names)
		self.X_test_names = sorted(self.X_test_names)
		
		assert self.label_file['image_name'].tolist() == self.X_train_names

	
	def read_image(self, img_name):
		if self.file_ext == 'tif' and self.bit_depth == 16:
			img = cv2.resize(cv2.imread(img_name, cv2.IMREAD_UNCHANGED), (self.image_base_size, self.image_base_size))
		else:
			img = misc.imresize(misc.imread(img_name), (self.image_base_size, self.image_base_size, self.channels))
		
		return np.array(img[:,:,:self.channels], dtype='float32')

	
	def load_images(self):
		if not os.path.exists(self.pickle_file_name):
			X_train = np.array([self.read_image(img_name) for img_name in self.train_img_names]) / self.max_image_value
			X_test = np.array([self.read_image(img_name) for img_name in self.test_img_names]) / self.max_image_value
			
			y_train = self.labels.toarray()
			
			self.data = dict()
			self.data['y_train'] = y_train
			self.data['X_train'] = X_train
			self.data['X_test'] = X_test
			
			with open(pickle_file, 'wb') as pickle_file:
				pickle_file.dump(self.data)
		else:
			with open(pickle_file, 'rb') as pickle_file:
				self.data = pickle.load(pickle_file)

	
	def load_images_mmap(self, validation_split=0.2):
		assert validation_split > 0.0
		
		num_test_imgs  = len(self.test_img_names)
		orig_num_train_imgs = len(self.train_img_names)
		num_valid_imgs = int(orig_num_train_imgs * validation_split)
		num_train_imgs = orig_num_train_imgs - num_valid_imgs
		
		print('Train: %d, Valid: %d, Test: %d' % (num_train_imgs, num_valid_imgs, num_test_imgs))
		
		if not (os.path.exists(self.X_train_mmap_file) and os.path.exists(self.X_test_mmap_file) and os.path.exists(self.X_valid_mmap_file)):
			train_imgs = np.arange(0, orig_num_train_imgs)
			valid_imgs = np.random.choice(train_imgs, num_valid_imgs, replace=False)
			train_imgs = np.setdiff1d(train_imgs, valid_imgs)
			
			self.images_to_mmap(list(np.array(self.train_img_names)[train_imgs]), self.X_train_mmap_file)
			self.images_to_mmap(list(np.array(self.train_img_names)[valid_imgs]), self.X_valid_mmap_file)
			self.images_to_mmap(self.test_img_names, self.X_test_mmap_file)
			
			y_train = self.labels.toarray()[train_imgs]
			np.save(self.y_train_file, y_train)
			
			y_valid = self.labels.toarray()[valid_imgs]
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
		return np.memmap(file_name, dtype='float32', mode = 'r', shape=(length, self.image_base_size, self.image_base_size, self.channels))
	
	
	def images_to_mmap(self, img_names, file_name):
		if os.path.exists(file_name):
			os.remove(file_name)
		
		print('Writing %d images to %s' % (len(img_names), file_name))
		
		os.mknod(file_name)
		data = np.memmap(file_name, dtype='float32', mode = 'r+', shape=(len(img_names), self.image_base_size, self.image_base_size, self.channels))
		
		with click.progressbar(range(len(img_names))) as vals:
			for i in vals:
				data[i] = np.array(self.read_image(img_names[i]) / self.max_image_value, dtype='float32')
				
				if i % 1000 == 0 and i > 0:
					data.flush()
		
		data.flush()
		data._mmap.close()
	
	
	def get_labels(self, y):
		y[y >= 0.5] = 1
		y[y < 0.5] = 0
		return np.array(self.vec.inverse_transform(y))

	
	def save_submission_file(self, y_pred):
		pred_labels = self.get_labels(y_pred)
		sub_file_name = os.path.join(self.base_dir, get_timestamp() + '_' + self.submission_file_name)
		
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

