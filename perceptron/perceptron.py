'''
Perceptron Class Implementation
Steps:
1. Initialize 
2. Compute activation u = w.T*x
3. Compute output y = signum(u + b) 
4. adapt weights w(n+1) = w(n) + lr*(d(n) - y)*x(n) 

This code also includes prediction steps to visualize the perceptron output

Author: Wesin Alves
'''
import numpy as np

class Perceptron(object):
	def __init__(self, timesteps, inputs, outputs, lr):
		self.timesteps = timesteps
		self.inputs = inputs
		self.outputs = outputs
		self.weights = np.zeros((inputs.shape[1], 1))
		self.train_samples = inputs.shape[0]
		self.lr = lr #learning rate

	def __signum(value):
		val = value[0]
		if val > 0:
			val = 1.0
		else:
			val = -1.0

		return val

	def predict(self, input_signal):
		v = self.weights.T.dot(input_signal)
		y = self.__signum(v)
		return y

	def adapt(self):
		for i in range(timesteps):
			mse = []
			errors = []
			for n in range(self.train_samples):
				y = predict(self.inputs[n,:])
				self.weights = self.weights + self.lr * (self.outputs[i] - y_n) * self.inputs[i,:]
				erros.append(self.outputs[i] - y_n)

			mse.append(np.mean(np.array(errors)**2))

		return mse


