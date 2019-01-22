import numpy as np
import matplotlib.pyplot as plt
# set variables
x_n = 2 * np.random.random(100) - 1
x_test = 2 * np.random.random(10) - 1

w_n = np.zeros(len(x_n))

y_n = np.zeros(len(x_n))
y_test = np.zeros(len(x_test))

line = 2*x_n + 0.3
line_test = 2*x_test + 0.3


neta = 0.3

timesteps = 10

def quantizer(vector, threshold):
	for i in range(len(vector)):
		if vector[i] > threshold:
			vector[i] = 1
		else:
			vector[i] = -1
	
	return vector

def sigum(value):
	if value > 0:
		value = 1
	else:
		value = -1

	return value

d_n = quantizer(line, np.mean(line))
d_test = quantizer(line_test, np.mean(line_test))

'''
debug >>>>>
print(x_n)
print(np.mean(line))
print(d_n)
'''
for n in range(len(x_n)-1):
	y_n[n] = sigum(np.dot(w_n.T[n],x_n[n]))
	w_n[n+1] = w_n[n] + neta*(d_n[n] - y_n[n])*x_n[n]

print(d_test)

mse = np.zeros(len(w_n))
for i in range(len(w_n)):
	for n in range(len(x_test)):
		y_test[n] = sigum(np.dot(w_n.T[i],x_test[n]))

	mse[i] = (np.sum(d_test - y_test)/len(y_test))**2


plt.plot(mse, label='mse')
plt.legend()
plt.show()