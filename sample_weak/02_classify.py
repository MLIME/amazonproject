import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
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
else:
    X_train_pca = np.load('X_train_pca.npy')
    X_test_pca = np.load('X_test_pca.npy')


#sum(pca.explained_variance_ratio_[0:128].tolist())
num_pcs = 128

rsvm = SVC(C = 1.0, kernel = 'rbf', probability=True)
psvm = SVC(C = 1.0, kernel = 'poly', degree=3, probability=True)
lsvm = LinearSVC(C = 1.0, loss='hinge')
rf = RandomForestClassifier()
xgb = XGBClassifier()

param_grid_svm = [{'C': [0.01, 0.1, 1.0]}]
param_grid_rf = [{'n_estimators': [100, 200, 400], 'max_features': [25, 50, 100, 128]}]

print('RandomForests')
grid_rf = GridSearchCV(rf, param_grid_rf, cv = 5, scoring = 'neg_mean_squared_error')
grid_rf.fit(X_train_pca[:,0:num_pcs], y_train)
print("Best RF: %f using %s" % (grid_rf.best_score_, grid_rf.best_params_))

print('R-SVM')
grid_rsvm = GridSearchCV(rsvm, param_grid_svm, cv = 5, scoring = 'neg_mean_squared_error')
grid_rsvm.fit(X_train_pca[:,0:num_pcs], y_train)
print("Best R-SVM: %f using %s" % (grid_rsvm.best_score_, grid_rsvm.best_params_))

print('L-SVM')
grid_lsvm = GridSearchCV(lsvm, param_grid_svm, cv = 5, scoring = 'neg_mean_squared_error')
grid_lsvm.fit(X_train_pca[:,0:num_pcs], y_train)
print("Best L-SVM: %f using %s" % (grid_lsvm.best_score_, grid_lsvm.best_params_))

print('P-SVM')
grid_psvm = GridSearchCV(psvm, param_grid_svm, cv = 5, scoring = 'neg_mean_squared_error')
grid_psvm.fit(X_train_pca[:,0:num_pcs], y_train)
print("Best P-SVM: %f using %s" % (grid_psvm.best_score_, grid_psvm.best_params_))

vc = VotingClassifier(estimators = [
        ('linear_svm', grid_lsvm.best_estimator_),
        ('radial_svm', grid_rsvm.best_estimator_),
        ('poly_svm', grid_psvm.best_estimator_),
        ('xgb', xgb),
        ('rf', grid_rf.best_estimator_)],
        voting='soft')

print('Voting Classifier')
vc.fit(X_train_pca[:,0:num_pcs], y_train)
y_pred = vc.predict(X_test_pca[:,0:num_pcs])
print(vc.__class__.__name__, accuracy_score(y_test, y_pred))

final_model = vc

print('Predicting on training data')
y_pred = final_model.predict_proba(X_train_pca)

pred_labels = get_labels(y_pred)
train_labels = get_labels(y_train)

pred_labels_list = []
train_labels_list = []

for i in range(len(y_pred)):
	print(label_file['image_name'][i] + ': ' + ','.join(train_labels[i]) + '|' + ','.join(pred_labels[i]))
	pred_labels_list.append(','.join(pred_labels[i]))
	train_labels_list.append(','.join(train_labels[i]))

label_file['train_labels'] = train_labels_list
label_file['pred_labels'] = pred_labels_list

label_file.to_csv('train_result.csv')

test_img_names = pd.read_csv('test_img_names.csv')

print('Predicting on test data')
y_pred = final_model.predict_proba(X_test_pca)
pred_labels = get_labels(y_pred)

submission_file = open('submission.csv', 'w')
submission_file.write('id,image_name,tags\n')

for i in range(n):
    id_file = test_img_names['image_name'][i].split('_')[1]
    s = id_file + ',' + test_img_names['image_name'][i] + ',' + ' '.join(pred_labels[i])
    print(s)
    submission_file.write(s)
    submission_file.write('\n')

submission_file.close()

submission_data = pd.read_csv('submission.csv')
submission_data = submission_data.sort_values('id').drop('id', 1)
submission_data.to_csv('submission.csv')

