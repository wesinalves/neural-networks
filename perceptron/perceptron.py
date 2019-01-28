import numpy as np
import matplotlib.pyplot as plt
# set variables
global x_n,x_tn, line, line_test

# Generate dataset
# data has gaussian distribution
n_samples = 100
mean_a = -1
mean_b = 1
sigma_a = 0.1
sigma_b = 0.1

x_1a = sigma_a * np.random.randn(n_samples//2) + mean_a
x_1b = sigma_b * np.random.randn(n_samples//2) + mean_b
x_1 = np.concatenate((x_1a,x_1b), axis=None)
#np.random.shuffle(x_1)

x_2a = sigma_a * np.random.randn(n_samples//2) + mean_a
x_2b = sigma_b * np.random.randn(n_samples//2) + mean_b
x_2 = np.concatenate((x_2a,x_2b), axis=None)
#np.random.shuffle(x_2)

x_t1 = sigma_a * np.random.randn(n_samples//3) + mean_a
x_t2 = sigma_b * np.random.randn(n_samples//3) + mean_b
#x_tn = np.array([x_t1,x_t2])

x_3 = x_1 + x_2

x_n = np.array([x_1,x_2,x_3])
x_n = np.random.permutation(x_n)
#print(x_n.shape)


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
w_n = np.zeros((2,1))
d_n = quantizer(x_3, 0)
lr = 0.1

'''
plt.figure(1)
plt.scatter(x_n[0],x_n[1], c=d_n)
plt.figure(2)
plt.plot(np.linspace(0,100,100),x_n[2])
plt.show()
'''

for i in range(n_samples):
	# compute activation
	u = w_n.T.dot(x_n[0:2,i])
	# compute response
	y_n = sigum(u)
	print(y_n, d_n[i])
	# adapt weights
	w_n = w_n + lr * ((d_n[i] - y_n)) * x_n[0:2,i]

diff = []
# compute mse
for i in range(n_samples):
	u = w_n.T.dot(x_n[0:2,i])
	y_n = sigum(u)
	#print(y_n, d_n[i])
	diff.append(d_n[i] - y_n)

mse = (np.sum(diff)**2) // len(diff)
print(mse)




'''
neta = 0.1

timesteps = 10

def quantizer(vector, threshold):
	x = []
	for val in vector:
		if val > threshold:
			x.append(1.0)
		else:
			x.append(-1.0)
	
	return x

def sigum(value):
	print(value)
	if value > 0:
		value = 1.0
	else:
		value = -1.0

	return value

d_n = quantizer(line, np.mean(line))
d_test = quantizer(line_test, np.mean(line_test))

cost = []

#for i in range(timesteps):
y_n = np.zeros(len(x_n))
for n in range(1,len(x_n)):
	y_n[n-1] = sigum( w_n.T.dot(x_n[:,n-1] ) )
	w_n = w_n + neta*(d_n[n-1] - y_n[n-1])*x_n[:,n-1].reshape(2,1)
	cost.append(np.sum( ((d_n[n-1] - y_n[n-1])**2)/2) )



for n in range(len(x_tn[0])-1):
	y_test[n] = sigum(w_n.T.dot(x_tn[:,n]))

mse = ((np.sum(d_test - y_test)**2)/len(y_test))

'''
'''
plt.figure(1)
plt.plot(cost, label='loss')
plt.figure(2)
plt.plot(line)
plt.legend()
plt.show()
'''

