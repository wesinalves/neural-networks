'''
Perceptron Implementation
Steps:
1. Initialize 
2. Compute activation u = w.T*x
3. Compute output y = signum(u + b) 
4. adapt weights w(n+1) = w(n) + lr*(d(n) - y)*x(n) 

This code also includes prediction steps to visualize the perceptron output

Author: Wesin Alves
'''
import numpy as np
import matplotlib.pyplot as plt
# set variables
global x_n,x_tn, line, line_test

# Generate dataset
# data has gaussian distribution
number_of_samples = 4	
mean_a = -1
mean_b = 1
sigma_a = 0.1
sigma_b = 0.1
timesteps = 2

x_1a = sigma_a * np.random.randn(number_of_samples//2) + mean_a
x_1b = sigma_b * np.random.randn(number_of_samples//2) + mean_b
x_1 = np.concatenate((x_1a,x_1b), axis=None)

x_2a = sigma_a * np.random.randn(number_of_samples//2) + mean_a
x_2b = sigma_b * np.random.randn(number_of_samples//2) + mean_b
x_2 = np.concatenate((x_2a,x_2b), axis=None)

x_t1 = sigma_a * np.random.randn(number_of_samples//3) + mean_a
x_t2 = sigma_b * np.random.randn(number_of_samples//3) + mean_b

x_3 = x_1 + x_2

b = np.ones(number_of_samples)

x_n = np.array([x_1,x_2, b])


# usefull functions
def quantizer(vector, threshold):
	x = []
	for val in vector:
		if val > threshold:
			x.append(1.0)
		else:
			x.append(-1.0)
	
	return x

def sigum(value):
	val = value[0]
	if val > 0:
		val = 1.0
	else:
		val = -1.0

	return val


# initialization
w_n = np.zeros((3,1))
d_n = quantizer(x_3, 0)
lr = 1


diff = []
for t in range(timesteps):
	mse = 0
	diff = []
	for i in range(number_of_samples):
		# compute activation
		u = w_n.T.dot(x_n[:,i])
		# compute response
		y_n = sigum(u)
		# adapt weights
		w_n = w_n + lr * ((d_n[i] - y_n)) * x_n[:,i]
		diff.append(np.fabs(d_n[i] - y_n))
	
	mse = (np.sum(diff)**2) / len(diff)
	print(mse)


# predictions
test = np.array([-1.1, -1.2, 1])
y_test = sigum(w_n.T.dot(test))
print(y_test)


