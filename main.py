#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import lib.importdir
import argparse
import itertools

lib.importdir.do("models", globals())
module_list = globals()['module_list']
module_list.remove('base_model') 

models = []

for module in module_list:
    mod = dir(globals()[module])
    
    for m in mod:
        if m not in ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']:
            klass = module + '.' + m
            baseClass = module + '.BaseModel'

            if 'Model' in m and issubclass(eval(klass), eval(baseClass)) and klass != baseClass:
                models.append(klass)

model_args = dict()

for model in models:
    m = eval(model)()
    model_args[model] = ','.join(['%s %s' % (k[0], k[1]) for k in m.args() or [None] if m.args() != None]) or ''


parser = argparse.ArgumentParser(description='Model Explorer')

parser.add_argument('base_dir', type=str,   help='base data directory')
parser.add_argument('train_dir', type=str,  help='train data directory')
parser.add_argument('test_dir', type=str,   help='test data directory')
parser.add_argument('model_name', type=str, help='model name' + '|'.join(['%s (args: %s)' % (k, v) for k, v in model_args.items()]))
parser.add_argument('--file_type', default='jpg', help='file type', choices=['jpg', 'tif'])
parser.add_argument('--labels', default='train.csv', help='label file name')
parser.add_argument('--img_size', type=int, default=256,  help='image size NxN (default: 256)')
parser.add_argument('--channels', type=int, default=3,  help='image channels (default: 3)')
parser.add_argument('--bit_depth', type=int, default=8, help='image bit depth (default: 8)')
parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 32)')
parser.add_argument('--num_epochs', type=int, default=100,  help='epochs (default: 100)')
parser.add_argument('--use_img_gen', type=bool, default=False, help='use image generator (default: False)')
parser.add_argument('--img_mult', type=int, default=4, help='when using image generator, multiples the training set size (default: 4)')
parser.add_argument('--saved_model_file', default=None, help='load saved model from file')

args, unknown = parser.parse_known_args()
arg_dict = dict(itertools.zip_longest(*[iter(unknown)] * 2, fillvalue=""))

base_dir = args.base_dir
train_dir = args.train_dir
test_dir = args.test_dir
model_name = args.model_name
file_type = args.file_type
label_file_name = args.labels
image_base_size = args.img_size
channels = args.channels
bit_depth = args.bit_depth
batch_size = args.batch_size
num_epochs = args.num_epochs
use_generator = args.use_img_gen
image_multiplier = args.img_mult
saved_model_file = args.saved_model_file

arg_dict['base_dir'] = args.base_dir
arg_dict['train_dir'] = args.train_dir
arg_dict['test_dir'] = args.test_dir
arg_dict['model_name'] = args.model_name
arg_dict['file_type'] = args.file_type
arg_dict['label_file_name'] = args.labels
arg_dict['image_base_size'] = args.img_size
arg_dict['channels'] = args.channels
arg_dict['bit_depth'] = args.bit_depth
arg_dict['batch_size'] = args.batch_size
arg_dict['num_epochs'] = args.num_epochs
arg_dict['use_generator'] = args.use_img_gen
arg_dict['image_multiplier'] = args.img_mult
arg_dict['saved_model_file'] = args.saved_model_file

if model_name not in models:
    print('Model %s not found' % model_name)
    exit(1)
else:
    model = eval(model_name)()


from lib.data_utils import DataManager

dm = DataManager(base_dir, model_name, train_dir, test_dir, file_type, image_base_size, channels, bit_depth, label_file_name)
dm.load_labels()
dm.load_file_list()
dm.load_images_mmap()

model.initialize(dm.num_categories, arg_dict)
model.fit(dm.data['X_train'], dm.data['y_train'], dm.data['X_valid'], dm.data['y_valid'])

y_pred = model.predict(dm.data['X_test'])
dm.save_submission_file(y_pred)
