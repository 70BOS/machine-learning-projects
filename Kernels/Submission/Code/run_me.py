# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

############################################################################
# Read in train and test synthetic data
def read_synthetic_data():
	print('Reading synthetic data ...')
	train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
	test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x, test_y)

############################################################################
# Read in train and test credit card data
def read_creditcard_data():
	print('Reading credit card data ...')
	train_x = np.loadtxt('../../Data/CreditCard/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/CreditCard/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/CreditCard/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Read in train and test tumor data
def read_tumor_data():
	print('Reading tumor data ...')
	train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter = ',', dtype=float)
	train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter = ',', dtype=float)
	test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter = ',', dtype=float)

	return (train_x, train_y, test_x)

############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
	# mean squared error
	return np.mean(np.power(y - y_hat, 2))

############################################################################
def kernelRidge(a,k,g,d,train_x,train_y):
    kf = KFold(n_splits=5)
    error = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        xiye = KernelRidge(alpha = a,kernel=k, gamma=g,degree=d)
        xiye.fit(x_train,y_train)
        y_pred = xiye.predict(x_test)
        error += compute_MSE(y_test,y_pred)
    return error/5
def svm(c,k,g,d,train_x,train_y):
    kf = KFold(n_splits=5)
    accuracy = 0.0
    for train_index, test_index in kf.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        clf = SVC(C=c, kernel=k, gamma=g,degree=d)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        accuracy += accuracy_score(y_test,y_pred)
    return accuracy/5
#return predicted y
def krrs(degree,kernel,train_x,train_y,test_x):
    if kernel=="poly":
        K = np.outer(train_x,train_x)
        for i in range(200):
            for j in range(200):
                K[i][j] = (K[i][j]+1)**degree
                if i==j:
                    K[i][j] += 0.1
        alpha = np.dot(np.linalg.inv(K),train_y)
    
        pred_y = []
        for x in np.nditer(test_x):
            s = 0.0
            for i in range(200):
                s += alpha[i]*((1+train_x[i]*x)**degree)
            pred_y.append(s)
        y = np.array(pred_y)
        return y
    elif kernel=="trig":
        K=np.ones((200,200))
        for i in range(200):
            for j in range(200):
                for k in range(1,degree+1):
                    shit=np.sin(0.5*k*train_x[i])*np.sin(0.5*k*train_x[j])+np.cos(0.5*k*train_x[i])*np.cos(0.5*k*train_x[j])
                    K[i][j] += shit
                if(i==j):
                    K[i][j] += 0.1
        alpha = np.dot(np.linalg.inv(K),train_y)
        pred_y = []
        for x in np.nditer(test_x):
            s = 0
            for i in range(200):
                kernel = 1
                for k in range(1,degree+1):
                    kernel+=np.sin(0.5*k*train_x[i])*np.sin(0.5*k*x)+np.cos(0.5*k*train_x[i])*np.cos(0.5*k*x)
                s += alpha[i]*kernel
            pred_y.append(s)
        y = np.array(pred_y)
        return y
    else:
        pass

def berr(degree,kernel,train_x,train_y,test_x):
    if kernel=="poly":
        expanded_x = np.zeros((len(train_x),degree+1))
        for i in range(len(train_x)):
            for j in range(degree+1):
                expanded_x[i][j]=train_x[i]**j
        
        clf = Ridge(alpha=0.1)
        clf.fit(expanded_x,train_y)
        expanded_test_x = np.zeros((len(test_x),degree+1))
        for i in range(len(test_x)):
            for j in range(degree+1):
                expanded_test_x[i][j]=test_x[i]**j
        return clf.predict(expanded_test_x)
    elif kernel=="trig":        
        x = []
        for i in range(len(train_x)):
            row = []
            for j in range(degree+1):               
                if j==0:
                    row.append(1)
                else:
                    row.append(np.sin(0.5*j*train_x[i]))
                    row.append(np.cos(0.5*j*train_x[i]))
            x.append(row)
        expanded_x = np.array(x)
        clf = Ridge(alpha=0.1)
        clf.fit(expanded_x,train_y)
        expanded_test_x = []
        for i in range(len(test_x)):
            row=[]
            for j in range(degree+1):               
                if j==0:
                    row.append(1)
                else:
                    row.append(np.sin(0.5*j*test_x[i]))
                    row.append(np.cos(0.5*j*test_x[i]))
            expanded_test_x.append(row)
        expanded_test_x=np.array(expanded_test_x)
        
        return clf.predict(expanded_test_x)
    else:
        pass

######################################################################
#Question 1d
#train_x, train_y, test_x, test_y = read_synthetic_data()
#print('Train=', train_x.shape)
#print('Test=', test_x.shape)
#
#y_poly_2 = krrs(2,"poly",train_x,train_y,test_x)
#y_poly_6 = krrs(6,"poly",train_x,train_y,test_x)
#y_trig_5 = krrs(5,"trig",train_x,train_y,test_x)
#y_trig_10 = krrs(10,"trig",train_x,train_y,test_x)
#
#berr_poly_2 = berr(2,"poly",train_x,train_y,test_x)
#berr_poly_6 = berr(6,"poly",train_x,train_y,test_x)
#berr_trig_5 = berr(5,"trig",train_x,train_y,test_x)
#berr_trig_10 = berr(10,"trig",train_x,train_y,test_x)
#
#
#plt.figure(figsize=(10,20)) 
#
#p1=plt.subplot(4, 2, 1)
#p1.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p1.scatter(test_x, y_poly_2, c='r', marker = "o",linewidth=1)
#p1.set_title("KRRS,Polynomial,degree=2,lambda=0.1")
#p1.set_xlabel("Test X") 
#p1.set_ylabel("True/Predicted Y") 
#
#p2=plt.subplot(4, 2, 2)
#p2.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p2.scatter(test_x, berr_poly_2, c='r', marker = "o",linewidth=1)
#p2.set_title("BERR,Polynomial,degree=2,lambda=0.1")
#p2.set_xlabel("Test X") 
#p2.set_ylabel("True/Predicted Y") 
#
#p3=plt.subplot(4, 2, 3)
#p3.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p3.scatter(test_x, y_poly_6, c='r', marker = "o",linewidth=1)
#p3.set_title("KRRS,Polynomial,degree=6,lambda=0.1")
#p3.set_xlabel("Test X") 
#p3.set_ylabel("True/Predicted Y") 
#
#p4=plt.subplot(4, 2, 4)
#p4.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p4.scatter(test_x, berr_poly_6, c='r', marker = "o",linewidth=1)
#p4.set_title("BERR,Polynomial,degree=6,lambda=0.1")
#p4.set_xlabel("Test X") 
#p4.set_ylabel("True/Predicted Y") 
#
#
#p5=plt.subplot(4, 2, 5)
#p5.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p5.scatter(test_x, y_trig_5, c='r', marker = "o",linewidth=1)
#p5.set_title("KRRS,Trigonometric,degree=5,lambda=0.1")
#p5.set_xlabel("Test X") 
#p5.set_ylabel("True/Predicted Y") 
#
#p6=plt.subplot(4, 2, 6)
#p6.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p6.scatter(test_x, berr_trig_5, c='r', marker = "o",linewidth=1)
#p6.set_title("BERR,Trigonometric,degree=5,lambda=0.1")
#p6.set_xlabel("Test X") 
#p6.set_ylabel("True/Predicted Y") 
#
#p7=plt.subplot(4, 2, 7)
#p7.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p7.scatter(test_x, y_trig_10, c='r', marker = "o",linewidth=1)
#p7.set_title("KRRS,Trigonometric,degree=10,lambda=0.1")
#p7.set_xlabel("Test X") 
#p7.set_ylabel("True/Predicted Y") 
#
#p8=plt.subplot(4, 2, 8)
#p8.scatter(test_x, test_y, c='b', marker = "*",linewidth=1)
#p8.scatter(test_x, berr_trig_10, c='r', marker = "o",linewidth=1)
#p8.set_title("BERR,Trigonometric,degree=10,lambda=0.1")
#p8.set_xlabel("Test X") 
#p8.set_ylabel("True/Predicted Y") 
#
##plt.savefig("../Figures/1d.pdf")
#plt.savefig("../Figures/1d.png")
#plt.show()
#
#y_poly_1 = krrs(1,"poly",train_x,train_y,test_x)
#y_poly_4 = krrs(4,"poly",train_x,train_y,test_x)
#y_trig_3 = krrs(3,"trig",train_x,train_y,test_x)
#berr_poly_1 = berr(1,"poly",train_x,train_y,test_x)
#berr_poly_4 = berr(4,"poly",train_x,train_y,test_x)
#berr_trig_3 = berr(3,"trig",train_x,train_y,test_x)
#
#krrsMSE = []
#berrMSE = []
#
#krrsMSE.append(compute_MSE(test_y, y_poly_1))
#krrsMSE.append(compute_MSE(test_y, y_poly_2))
#krrsMSE.append(compute_MSE(test_y, y_poly_4))
#krrsMSE.append(compute_MSE(test_y, y_poly_6))
#krrsMSE.append(compute_MSE(test_y, y_trig_3))
#krrsMSE.append(compute_MSE(test_y, y_trig_5))
#krrsMSE.append(compute_MSE(test_y, y_trig_10))
#
#berrMSE.append(compute_MSE(test_y, berr_poly_1))
#berrMSE.append(compute_MSE(test_y, berr_poly_2))
#berrMSE.append(compute_MSE(test_y, berr_poly_4))
#berrMSE.append(compute_MSE(test_y, berr_poly_6))
#berrMSE.append(compute_MSE(test_y, berr_trig_3))
#berrMSE.append(compute_MSE(test_y, berr_trig_5))
#berrMSE.append(compute_MSE(test_y, berr_trig_10))
#
#print("krrs MSE: ",krrsMSE)
#print("berr MSE: ",berrMSE)
#
##Question 1e
#train_x, train_y, test_x  = read_creditcard_data()
#print('Train=', train_x.shape)
#print('Test=', test_x.shape)
#Errors = []
#parameters = [(1,"rbf",None,0),(1,"poly",None,3),(1,"linear",None,0),
#              (1,"rbf",1,0),(1,"poly",1,3),
#              (1,"rbf",0.001,0),(1,"poly",0.001,3),
#              (0.0001,"rbf",None,0),(0.0001,"poly",None,3),(0.0001,"linear",None,0),
#              (0.0001,"rbf",1,0),(0.0001,"poly",1,3),
#              (0.0001,"rbf",0.001,0),(0.0001,"poly",0.001,3)]
#for (a,k,g,d) in parameters:
#    Errors.append(kernelRidge(a,k,g,d,train_x,train_y))
#print(Errors)
#print("min mse: ",min(Errors))
#(a,k,g,d) = parameters[Errors.index(min(Errors))]
#print("model: ",a,k,g,d)
#bestKRR = KernelRidge(alpha = a,kernel=k,gamma=g,degree=d)
#bestKRR.fit(train_x,train_y)
#predicted_y = bestKRR.predict(test_x)
#file_name = '../Predictions/CreditCard/best.csv'
#print('Writing output to ', file_name)
#kaggle.kaggleize(predicted_y, file_name, True)


#Question 2
train_x, train_y, test_x  = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)
Accuracy = []
parameters = [(1,"rbf",1,3),(1,"poly",1,3),(1,"poly",1,5),(1,"linear",1,3),
              (1,"rbf",0.01,3),(1,"poly",0.01,3),(1,"poly",0.01,5),
              (1,"rbf",0.001,3),(1,"poly",0.001,3),(1,"poly",0.001,5),
              (0.01,"rbf",1,3),(0.01,"poly",1,3),(0.01,"poly",1,5),(0.01,"linear",1,3),
              (0.01,"rbf",0.01,3),(0.01,"poly",0.01,3),(0.01,"poly",0.01,5),
              (0.01,"rbf",0.001,3),(0.01,"poly",0.001,3),(0.01,"poly",0.001,5),
              (0.0001,"rbf",1,3),(0.0001,"poly",1,3),(0.0001,"poly",1,5),(0.0001,"linear",1,3),
              (0.0001,"rbf",0.01,3),(0.0001,"poly",0.01,3),(0.0001,"poly",0.01,5),
              (0.0001,"rbf",0.001,3),(0.0001,"poly",0.001,3),(0.0001,"poly",0.001,5)]
for (c,k,g,d) in parameters:
    Accuracy.append(svm(c,k,g,d,train_x,train_y))
(c,k,g,d) = parameters[Accuracy.index(max(Accuracy))]
bestSVM = SVC(C=c, kernel=k, gamma=g,degree=d)
print("Accuracy: ",Accuracy)
print("max accuracy: ", max(Accuracy))
print("model: ",c,k,g,d)
bestSVM.fit(train_x,train_y)
predicted_y = bestSVM.predict(test_x)

file_name = '../Predictions/Tumor/best.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name, False)



#################OWARI
## Create dummy test output values to compute accuracy
#test_y = np.random.randint(0, 2, (test_x.shape[0], 1))
#predicted_y = np.random.randint(0, 2, (test_x.shape[0], 1))
#print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))
#
## Output file location
#file_name = '../Predictions/Tumor/best.csv'
## Writing output in Kaggle format
#print('Writing output to ', file_name)
#kaggle.kaggleize(predicted_y, file_name, False)

