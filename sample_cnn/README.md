# Sample CNN using Keras

**Beware of alpha-stage code**

* 01_convert_data.py
Script to convert the image files into numpy arrays. You must copy the "test-jpg" and "train-jpg" folders into the directory Data before running (a script for doing so will be provided soon !!).

The variable "image_base_size" determines the size of the image to be used by the CNN, as an int value. Original image size is 256x256, and the script uses 96x96 by default to overcome OOM problems when using the GPU.

Data will be converted into 3 numpy arrays: X_train.npy, X_test.npy and y_train.npy.

* 02_classify.py
Script to train/test the image classification model based on CNN. There are two modes of execution: train/test and test-only.

  - Test-only
  A pre-trained model is available on "final_model.h5", and it will be loaded by default, so the script can run the test phase directly.

  - Train/test
  If you want to train a new model, rename "final_model.h5" to anything else, and the script will start training a new model, that will be saved to "final_model.h5" at the end. This can take a long time.

* Training tunables

The _create_model_ function defines the CNN architecture.

The following lists can have values added or removed to allow grid search to run several models and pick the best one at the end. Be careful when adding values, because it can result in a very large number of models to be trained.

```python
	optimizers = ['adam']
	inits = ['he_uniform']
	epochs = [50]
	batch_sizes = [24]
	window_sizes = [7]
	hidden_layer_sizes = [64]
	dropouts1 = [0.2]
	dropouts2 = [0.5]
	activations = ['relu']
```

The validation split (fraction of training data that will be used on validation) can be changed on the fit_params dictionary:

```python
fit_params = dict(validation_split=0.3,
				  verbose=1,
				  callbacks=callbacks)
```

The number of CV folds can be changed on the GridSearchCV constructor call:

```python
grid = GridSearchCV(estimator=model, param_grid=param_grid, fit_params=fit_params, cv=3, verbose=1)
```
