import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

np.random.seed(777)

image_base_size = 28

label_file_name = 'train.csv'
label_file = pd.read_csv(label_file_name)
label_file = label_file.sort_values('image_name')

vec = CountVectorizer(min_df=1)
labels = vec.fit_transform(label_file['tags'])

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')

num_categories = y_train.shape[1]

def get_labels(y):
    y[y >= 0.5] = True
    y[y < 0.5] = False
    return np.array(vec.inverse_transform(y))


X_train = X_train.reshape(len(X_train), image_base_size*image_base_size*4)
X_test = X_test.reshape(len(X_test), image_base_size*image_base_size*4)

if not os.path.exists('X_train_pca.npy'):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    np.save('X_train_pca.npy', X_train_pca)
    np.save('X_test_pca.npy', X_test_pca)
    
    for i in range(X_train_pca.shape[1]):
        var_explained = sum(pca.explained_variance_ratio_[0:i].tolist())
        print('%d variables, %.2f variance explained' % (i, var_explained))
        if var_explained >= 0.95:
            break
    exit()
else:
    X_train_pca = np.load('X_train_pca.npy')
    X_test_pca = np.load('X_test_pca.npy')

num_pcs = 32

param_grid_rf = [{'n_estimators': [100, 200, 400], 'max_features': [10, 20, 32]}]

rf = RandomForestClassifier()

print('RandomForests')
grid_rf = GridSearchCV(rf, param_grid_rf, cv = 5, scoring = 'neg_mean_squared_error', verbose=1)
grid_rf.fit(X_train_pca[:,0:num_pcs], y_train)
print("Best RF: %f using %s" % (grid_rf.best_score_, grid_rf.best_params_))

final_model = grid_rf.best_model_

print('Predicting on test data')
y_pred = final_model.predict_proba(X_test)
pred_labels = get_labels(y_pred)

submission_file = open('submission.csv', 'w')
submission_file.write('id,image_name,tags\n')

for i in range(len(pred_labels)):
    id_file = test_img_names['image_name'][i].split('_')[1].replace('.jpg', '')
    s = id_file + ',' + os.path.basename(test_img_names['image_name'][i]).replace('.jpg', '') + ',' + ' '.join(pred_labels[i])
    print(s)
    submission_file.write(s)
    submission_file.write('\n')

submission_file.close()

submission_data = pd.read_csv('submission.csv')
submission_data = submission_data.sort_values('id').drop('id', 1)
submission_data.to_csv('submission.csv', index=False)
