# Import python modules
import numpy as np
import kaggle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import time

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
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

#return average MAE
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

#return average MAE
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

#return average MAE
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

#return average MAE
def dt(depth,train_x,train_y,fold):
    kf = KFold(n_splits=fold)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        pdd = DecisionTreeRegressor(max_depth = depth)
        pdd.fit(x_train,y_train)
        y_hat = pdd.predict(x_test)
        error += compute_error(y_hat,y_test)
    return error/5


##########################POWER PLANT#################################

train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('YTest=', train_y.shape)
print('Test=', test_x.shape)

depth = [3,6,9,12,15]
k = [3,5,10,20,25]
alpha = [1e-6,1e-4,1e-2,1,10]
dtErrors = []
knnErrors = []   
lassoErrors = [] 
ridgeErrors = []
 
#-----------------decision tree-------------------#
print("training decision tree models ...")
timeElapse = []

start = time.time()
dtErrors.append(dt(3,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(6,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(9,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(12,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(15,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

X_LABEL = "depth"
Y_LABEL = "time"
plt.scatter(depth, timeElapse)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.show()

print("depth",depth)
print("predicted errors ",dtErrors)
print()
#---------------------knn--------------------------#
print("training knn models ...")

knnErrors.append(knn(3,train_x,train_y))
knnErrors.append(knn(5,train_x,train_y))
knnErrors.append(knn(10,train_x,train_y))
knnErrors.append(knn(20,train_x,train_y))
knnErrors.append(knn(25,train_x,train_y))

print("k ",k)
print("predicted errors ",knnErrors)
print()
#--------------------lasso------------------------#
print("training lasso regression models ...")

lassoErrors.append(lasso(1e-6,train_x,train_y))
lassoErrors.append(lasso(1e-4,train_x,train_y))
lassoErrors.append(lasso(1e-2,train_x,train_y))
lassoErrors.append(lasso(1,train_x,train_y))
lassoErrors.append(lasso(10,train_x,train_y))

print("alpha ",alpha)
print("predicted errors ",lassoErrors)
print()
#--------------------ridge------------------------#
print("training ridge regression models ...")

ridgeErrors.append(ridge(1e-6,train_x,train_y))
ridgeErrors.append(ridge(1e-4,train_x,train_y))
ridgeErrors.append(ridge(1e-2,train_x,train_y))
ridgeErrors.append(ridge(1,train_x,train_y))
ridgeErrors.append(ridge(10,train_x,train_y))

print("alpha",ridgeErrors)
print("predicted errors ",ridgeErrors)
print()
#---------------writing output-------------------#
d = depth[dtErrors.index(min(dtErrors))]
decisionTree = DecisionTreeRegressor(max_depth=d)
decisionTree.fit(train_x,train_y)
predicted_y = decisionTree.predict(test_x)
file_name = '../Predictions/PowerOutput/dtbest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

neighbors = k[knnErrors.index(min(knnErrors))]
KNN = KNeighborsRegressor(n_neighbors = neighbors)
KNN.fit(train_x,train_y)
predicted_y = KNN.predict(test_x)
file_name = '../Predictions/PowerOutput/knnBest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

a = alpha[lassoErrors.index(min(lassoErrors))]
lassoRegressor = Lasso(alpha = a)
lassoRegressor.fit(train_x,train_y)
predicted_y = lassoRegressor.predict(test_x)
file_name = '../Predictions/PowerOutput/lassoBest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

a = alpha[ridgeErrors.index(min(ridgeErrors))]
ridgeRegressor = Ridge(alpha = a)
ridgeRegressor.fit(train_x,train_y)
predicted_y = ridgeRegressor.predict(test_x)
file_name = '../Predictions/PowerOutput/ridgeBest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)


###########INDOOR LOCATION ##############

train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

depth2 = [20,25,30,35,40]
alpha = [1e-4,1e-2,1,10]
dtErrors.clear()
knnErrors.clear()  
lassoErrors.clear() 
ridgeErrors.clear()

#-----------------decision tree-------------------#
print("training decision tree models ...")
timeElapse.clear()

start = time.time()
dtErrors.append(dt(20,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(25,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(30,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(35,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)

start = time.time()
dtErrors.append(dt(40,train_x,train_y,5))
end = time.time()
timeElapse.append((end-start)*1000)


X_LABEL = "depth"
Y_LABEL = "time"
plt.scatter(depth2, timeElapse)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.show()

print("depth",depth2)
print("predicted errors ",dtErrors)

d = depth[dtErrors.index(min(dtErrors))]
decisionTree = DecisionTreeRegressor(max_depth=d)
decisionTree.fit(train_x,train_y)
predicted_y = decisionTree.predict(test_x)
file_name = '../Predictions/IndoorLocalization/dtbest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print()

#---------------------knn--------------------------#
print("training knn models ...")

knnErrors.append(knn(3,train_x,train_y))
knnErrors.append(knn(5,train_x,train_y))
knnErrors.append(knn(10,train_x,train_y))
knnErrors.append(knn(20,train_x,train_y))
knnErrors.append(knn(25,train_x,train_y))

print("k ",k)
print("predicted errors ",knnErrors)

neighbors = k[knnErrors.index(min(knnErrors))]
KNN = KNeighborsRegressor(n_neighbors = neighbors)
KNN.fit(train_x,train_y)
predicted_y = KNN.predict(test_x)
file_name = '../Predictions/IndoorLocalization/knnBest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print()

#--------------------lasso------------------------#
print("training lasso regression models ...")

lassoErrors.append(lasso(1e-4,train_x,train_y))
lassoErrors.append(lasso(1e-2,train_x,train_y))
lassoErrors.append(lasso(1,train_x,train_y))
lassoErrors.append(lasso(10,train_x,train_y))

print("alpha ",alpha)
print("predicted errors ",lassoErrors)

a = alpha[lassoErrors.index(min(lassoErrors))]
lassoRegressor = Lasso(alpha = a)
lassoRegressor.fit(train_x,train_y)
predicted_y = lassoRegressor.predict(test_x)
file_name = '../Predictions/IndoorLocalization/lassoBest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print()

#--------------------ridge------------------------#
print("training ridge regression models ...")

ridgeErrors.append(ridge(1e-4,train_x,train_y))
ridgeErrors.append(ridge(1e-2,train_x,train_y))
ridgeErrors.append(ridge(1,train_x,train_y))
ridgeErrors.append(ridge(10,train_x,train_y))

print("alpha ",alpha)
print("predicted errors ",lassoErrors)

a = alpha[ridgeErrors.index(min(ridgeErrors))]
ridgeRegressor = Ridge(alpha = a)
ridgeRegressor.fit(train_x,train_y)
predicted_y = ridgeRegressor.predict(test_x)
file_name = '../Predictions/IndoorLocalization/ridgeBest.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print()







#-----------------Question 5a--------------------#
depth = [8,9,10]
dtErrors = []

dtErrors.append(dt(8,train_x,train_y,10))
dtErrors.append(dt(9,train_x,train_y,10))
dtErrors.append(dt(10,train_x,train_y,10))

print("index, minimum predicted error: ",dtErrors.index(min(dtErrors)),", ",min(dtErrors))


d = depth[dtErrors.index(min(dtErrors))]
decisionTree = DecisionTreeRegressor(max_depth=d)
decisionTree.fit(train_x,train_y)
predicted_y = decisionTree.predict(test_x)
file_name = '../Predictions/PowerOutput/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print()
#-----------------Question 5b--------------------#
k = [1,2,3,4]
knnErrors =[]  

knnErrors.append(knn(1,train_x,train_y))
knnErrors.append(knn(2,train_x,train_y))
knnErrors.append(knn(3,train_x,train_y))
knnErrors.append(knn(4,train_x,train_y))

print("index, minimum predicted error: ", knnErrors.index(min(knnErrors)),", ",min(knnErrors))

neighbors = k[knnErrors.index(min(knnErrors))]
KNN = KNeighborsRegressor(n_neighbors = neighbors)
KNN.fit(train_x,train_y)
predicted_y = KNN.predict(test_x)
file_name = '../Predictions/IndoorLocalization/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)









