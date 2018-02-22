# Import python modules
import numpy as np
import kaggle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
#import matplotlib.pyplot as plt

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):#传入两个numpy array
	# mean absolute error
	return np.abs(y_hat - y).mean()

#return MAE on y_test
def knn(k,train_x,train_y):
    kf = KFold(n_splits=5)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        pdd = KNeighborsRegressor(n_neighbors = k)
        pdd.fit(x_train,y_train)
        y_hat = pdd.predict(x_test)
        error += compute_error(y_hat,y_test)
    return error/5

def lasso(a,train_x,train_y):
    kf = KFold(n_splits=5)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        pdd = Lasso(alpha = a)
        pdd.fit(x_train,y_train)
        y_hat = pdd.predict(x_test)
        error += compute_error(y_hat,y_test)
    return error/5

def ridge(a,train_x,train_y):
    kf = KFold(n_splits=5)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        pdd = Ridge(alpha = a)
        pdd.fit(x_train,y_train)
        y_hat = pdd.predict(x_test)
        error += compute_error(y_hat,y_test)
    return error/5

def dt(depth,train_x,train_y):
    kf = KFold(n_splits=5)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        pdd = DecisionTreeRegressor(max_depth = depth)
        pdd.fit(x_train,y_train)
        y_hat = pdd.predict(x_test)
        error += compute_error(y_hat,y_test)
    return error/5
#########################################################################

train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('YTest=', train_y.shape)
print('Test=', test_x.shape)

    ############# Music S.T.A.R.T!! ###############
depth = [3,6,9,12,15]
k = [3,5,10,20,25]
alpha = [1e-6,1e-4,1e-2,1,10]
dtErrors = []
knnErrors = []   
lassoErrors = [] 
ridgeErrors = [] 

dtErrors.append(dt(3,train_x,train_y))
dtErrors.append(dt(6,train_x,train_y))
dtErrors.append(dt(9,train_x,train_y))
dtErrors.append(dt(12,train_x,train_y))
dtErrors.append(dt(15,train_x,train_y))

knnErrors.append(knn(3,train_x,train_y))
knnErrors.append(knn(5,train_x,train_y))
knnErrors.append(knn(10,train_x,train_y))
knnErrors.append(knn(20,train_x,train_y))
knnErrors.append(knn(25,train_x,train_y))

lassoErrors.append(lasso(1e-6,train_x,train_y))
lassoErrors.append(lasso(1e-4,train_x,train_y))
lassoErrors.append(lasso(1e-2,train_x,train_y))
lassoErrors.append(lasso(1,train_x,train_y))
lassoErrors.append(lasso(10,train_x,train_y))
   
ridgeErrors.append(ridge(1e-6,train_x,train_y))
ridgeErrors.append(ridge(1e-4,train_x,train_y))
ridgeErrors.append(ridge(1e-2,train_x,train_y))
ridgeErrors.append(ridge(1,train_x,train_y))
ridgeErrors.append(ridge(10,train_x,train_y))

print(dtErrors.index(min(dtErrors)))
print(knnErrors.index(min(knnErrors)))
print(lassoErrors.index(min(lassoErrors)))
print(ridgeErrors.index(min(ridgeErrors)))

# Create dummy test output values
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)



###############indoor##############
train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', test_x.shape)
# Create dummy test output values
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

