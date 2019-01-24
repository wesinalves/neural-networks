import numpy as np
import matplotlib.pyplot as plt
# set variables
global x_n,x_test, line, line_test

# data has gaussian distribution
n_samples = 100
mean_a = -1
mean_b = 1
sigma_a = 0.3
sigma_b = 0.3

x_n = np.concatenate([(sigma_a * np.random.randn(n_samples//2) + mean_a), (sigma_b * np.random.randn(n_samples//2) + mean_b) ], axis=None)
x_test = np.concatenate([(sigma_a * np.random.randn(n_samples//3) + mean_a), (sigma_b * np.random.randn(n_samples//3) + mean_b) ], axis=None)

w_n = np.zeros(len(x_n))

y_n = np.zeros(len(x_n))
y_test = np.zeros(len(x_test))

line = 2*x_n + 0.3
line_test = 2*x_test + 0.3



neta = 0.3

timesteps = 10

def quantizer(vector, threshold):
	x = []
	for val in vector:
		if val > threshold:
			x.append(1)
		else:
			x.append(-1)
	
	return x

def sigum(value):
	if value > 0:
		value = 1
	else:
		value = -1

	return value

d_n = quantizer(line, np.mean(line))
d_test = quantizer(line_test, np.mean(line_test))


for n in range(len(x_n)-1):
	y_n[n] = sigum(np.dot(w_n.T[n],x_n[n]))
	w_n[n+1] = w_n[n] + neta*(d_n[n] - y_n[n])*x_n[n]


mse = np.zeros(len(w_n))
for i in range(len(w_n)):
	for n in range(len(x_test)):
		y_test[n] = sigum(np.dot(w_n.T[i],x_test[n]))

	mse[i] = (np.sum(d_test - y_test)/len(y_test))**2

plt.figure(1)
plt.plot(mse, label='mse')
plt.figure(2)
plt.scatter(x_n, line)
plt.legend()
plt.show()