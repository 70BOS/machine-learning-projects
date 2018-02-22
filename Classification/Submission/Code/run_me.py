# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

# Read in train and test data
def read_image_data():
	print('Reading image data ...')
	temp = np.load('../../Data/data_train.npz')
	train_x = temp['data_train']
	temp = np.load('../../Data/labels_train.npz')
	train_y = temp['labels_train']
	temp = np.load('../../Data/data_test.npz')
	test_x = temp['data_test']
	return (train_x, train_y, test_x)

#return estimated error of this model
def DT(depth,train_x,train_y,fold):
    kf = KFold(n_splits=fold)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        pdd = DecisionTreeClassifier(max_depth = depth)
        pdd.fit(x_train,y_train)
        y_pred = pdd.predict(x_test)
        error += 1-accuracy_score(y_test,y_pred)
    return error/fold

def knn(k,train_x,train_y,fold):
    kf = KFold(n_splits=fold)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        xiye = KNeighborsClassifier(n_neighbors=k)
        xiye.fit(x_train,y_train)
        y_pred = xiye.predict(x_test)
        error += 1-accuracy_score(y_test,y_pred)
    return error/fold

def linearModel(l,a,train_x,train_y,fold):
    kf = KFold(n_splits=fold)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        condi = SGDClassifier(loss=l, penalty='l2',alpha = a)
        condi.fit(x_train,y_train)
        y_pred = condi.predict(x_test)
        error += 1-accuracy_score(y_test,y_pred)
    return error/fold
############################################################################
train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

#---Decision Tree---#

depth = [3,6,9,12,14]
dtErrors = []
dtErrors.append(DT(3,train_x,train_y,5))
dtErrors.append(DT(6,train_x,train_y,5))
dtErrors.append(DT(9,train_x,train_y,5))
dtErrors.append(DT(12,train_x,train_y,5))
dtErrors.append(DT(14,train_x,train_y,5))
print(dtErrors)
d = depth[dtErrors.index(min(dtErrors))]
decisionTree = DecisionTreeClassifier(max_depth=d)
decisionTree.fit(train_x,train_y)
predicted_y = decisionTree.predict(test_x)
file_name = '../Predictions/dtbest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

#---KNN---#
n = [3,5,7,9,11]
knnErrors = []
knnErrors.append(knn(3,train_x,train_y,5))
knnErrors.append(knn(5,train_x,train_y,5))
knnErrors.append(knn(7,train_x,train_y,5))
knnErrors.append(knn(9,train_x,train_y,5))
knnErrors.append(knn(11,train_x,train_y,5))

print(knnErrors)
k = n[knnErrors.index(min(knnErrors))]
bestKnnModel = KNeighborsClassifier(n_neighbors=k)
bestKnnModel.fit(train_x,train_y)
predicted_y = bestKnnModel.predict(test_x)
file_name = '../Predictions/knnbest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

##---Linear Model---#
alpha = [1e-6,1e-4,1e-2,1,10,1e-6,1e-4,1e-2,1,10]
lmErrors = []
lmErrors.append(linearModel('hinge',1e-6,train_x,train_y,5))
lmErrors.append(linearModel('hinge',1e-4,train_x,train_y,5))
lmErrors.append(linearModel('hinge',1e-2,train_x,train_y,5))
lmErrors.append(linearModel('hinge',1,train_x,train_y,5))
lmErrors.append(linearModel('hinge',10,train_x,train_y,5))

lmErrors.append(linearModel('log',1e-6,train_x,train_y,5))
lmErrors.append(linearModel('log',1e-4,train_x,train_y,5))
lmErrors.append(linearModel('log',1e-2,train_x,train_y,5))
lmErrors.append(linearModel('log',1,train_x,train_y,5))
lmErrors.append(linearModel('log',10,train_x,train_y,5))

print(lmErrors)
bestAlpha = alpha[lmErrors.index(min(lmErrors))]
print(bestAlpha)
if(bestAlpha/5>=1):
    l = 'log'
else:
    l = 'hinge'
bestLM = SGDClassifier(loss=l, penalty='l2',alpha = bestAlpha)
bestLM.fit(train_x,train_y)
predicted_y = bestLM.predict(test_x)
file_name = '../Predictions/LMbest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)




