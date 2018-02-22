import autograd.numpy as np
import autograd
from autograd.util import flatten
import matplotlib.pyplot as plt
import time
import kaggle
from sklearn.model_selection import train_test_split

# Function to compute classification accuracy
def mean_zero_one_loss(weights, x, y_integers, unflatten):
	(W, b, V, c) = unflatten(weights)
	out = feedForward(W, b, V, c, x)
	pred = np.argmax(out, axis=1)
	return(np.mean(pred != y_integers))

# Feed forward output i.e. L = -O[y] + log(sum(exp(O[j])))
def feedForward(W, b, V, c, train_x):
        hid = np.tanh(np.dot(train_x, W) + b)
        out = np.dot(hid, V) + c
        return out

# Logistic Loss function
def logistic_loss_batch(weights, x, y, unflatten):
	# regularization penalty
        lambda_pen = 10

        # unflatten weights into W, b, V and c respectively 
        (W, b, V, c) = unflatten(weights)

        # Predict output for the entire train data
        out  = feedForward(W, b, V, c, x)
        pred = np.argmax(out, axis=1)

        # True labels
        true = np.argmax(y, axis=1)
        # Mean accuracy
        class_err = np.mean(pred != true)

        # Computing logistic loss with l2 penalization
        logistic_loss = np.sum(-np.sum(out * y, axis=1) + np.log(np.sum(np.exp(out),axis=1))) + lambda_pen * np.sum(weights**2)
        
        # returning loss. Note that outputs can only be returned in the below format
        return (logistic_loss, [autograd.util.getval(logistic_loss), autograd.util.getval(class_err)])

def question4b3(m,train_x,train_y_integers):
    # Number of hidden units
    dims_hid = m
    # Compress all weights into one weight vector using autograd's flatten
    x_train, x_test, y_train_integers, y_test_integers = train_test_split(train_x, train_y_integers, test_size=0.2,train_size=0.8)
    y_train = np.zeros((x_train.shape[0], 4))
    y_train[np.arange(x_train.shape[0]), y_train_integers] = 1
    y_test = np.zeros((x_test.shape[0], 4))
    y_test[np.arange(x_test.shape[0]), y_test_integers] = 1
    
    W = np.random.randn(x_train.shape[1], dims_hid)
    b = np.random.randn(dims_hid)
    V = np.random.randn(dims_hid, 4)
    c = np.random.randn(4)

    all_weights = (W, b, V, c)
    weights, unflatten = flatten(all_weights)
    smooth_grad = 0

    for i in range(1000):
        weight_gradients, returned_values = grad_fun(weights, x_train, y_train, unflatten)
        smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
        weights = weights - epsilon * smooth_grad

    return mean_zero_one_loss(weights, x_test, y_test_integers, unflatten)
####################################################################################
# Loading the dataset
print('Reading image data ...')
temp = np.load('../../Data/data_train.npz')
train_x = temp['data_train']
temp = np.load('../../Data/labels_train.npz')
train_y_integers = temp['labels_train']
temp = np.load('../../Data/data_test.npz')
test_x = temp['data_test']

# Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
train_x -= .5
test_x  -= .5

# Number of output dimensions
dims_out = 4
# Number of hidden units
dims_hid = 5
# Learning rate
epsilon = 0.0001
# Momentum of gradients update
momentum = 0.1
# Number of epochs
nEpochs = 1000
# Number of train examples
nTrainSamples = train_x.shape[0]
# Number of input dimensions
dims_in = train_x.shape[1]

# Convert integer labels to one-hot vectors
# i.e. convert label 2 to 0, 0, 1, 0
train_y = np.zeros((nTrainSamples, dims_out))
train_y[np.arange(nTrainSamples), train_y_integers] = 1

assert momentum <= 1
assert epsilon <= 1

# Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
grad_fun = autograd.grad_and_aux(logistic_loss_batch)

# Initializing weights
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0

# Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)
trainingTime = []
meanloss5 = []
meanloss40 = []
meanloss70 = []

start = time.time()
for i in range(nEpochs):
    # Compute gradients (partial derivatives) using autograd toolbox
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    # Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
    meanloss5.append(returned_values[0]/20000)
    
end = time.time()
trainingTime.append((end-start)*1000)
print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))


#####40
dims_hid = 40
# Initializing weights
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0

# Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)


start = time.time()
for i in range(nEpochs):
    # Compute gradients (partial derivatives) using autograd toolbox
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    # Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
    meanloss40.append(returned_values[0]/20000)
    
end = time.time()
trainingTime.append((end-start)*1000)
print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))

#########70
dims_hid = 70
# Initializing weights
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0

# Compress all weights into one weight vector using autograd's flatten
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)


start = time.time()
for i in range(nEpochs):
    # Compute gradients (partial derivatives) using autograd toolbox
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    # Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
    meanloss70.append(returned_values[0]/20000)
    
end = time.time()
trainingTime.append((end-start)*1000)
print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))
print('training time: ',trainingTime)

inds = np.arange(1000)
labels = ["M = 5","M = 40","M = 70"]
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,meanloss5,'r-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,meanloss40,'b-', linewidth=3) #Plot the first series in blue with square marker
plt.plot(inds,meanloss70,'g-', linewidth=3)
plt.grid(True) #Turn the grid on
plt.ylabel("mean training logistic loss") #Y-axis label
plt.xlabel("epochs") #X-axis label
plt.xlim(0,1000) #set x axis range
plt.ylim(0,50) #Set yaxis range
plt.legend(labels,loc="best")
plt.savefig("../Figures/4bplot.pdf")
plt.show()


###############################################Question 4b3###################################
loss = []
xiye = [5,40,70]
loss.append(question4b3(5,train_x,train_y_integers))
loss.append(question4b3(40,train_x,train_y_integers))
loss.append(question4b3(70,train_x,train_y_integers))
print('loss: ',loss)
dims_hid = xiye[loss.index(min(loss))]
# Initializing weights
W = np.random.randn(dims_in, dims_hid)
b = np.random.randn(dims_hid)
V = np.random.randn(dims_hid, dims_out)
c = np.random.randn(dims_out)
smooth_grad = 0
all_weights = (W, b, V, c)
weights, unflatten = flatten(all_weights)
for i in range(nEpochs):
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
    
(W, b, V, c) = unflatten(weights)
out  = feedForward(W, b, V, c, test_x)
pred = np.argmax(out, axis=1)
file_name = '../Predictions/neuralnetwork.csv'
print('Writing output to ', file_name)
kaggle.kaggleize(pred, file_name)
#score 0.26614(kaggle accuracy)