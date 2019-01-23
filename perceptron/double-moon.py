import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class DoubleMoon(object):
	"""docstring for DoubleMoon"""
	def __init__(self, w, r, d, n):
		super(DoubleMoon, self).__init__()
		self.w = w
		self.r = r
		self.d = d
		self.n = n

	def get_values(self):
		n_out = self.n // 2
		n_in = self.n // 2

		# generator = check_random_state(random_state) sklearn method to check the seed parameter

		outer_circ_x = 2* np.cos(np.linspace(0, np.pi, n_out))
		outer_circ_y = 2* np.sin(np.linspace(0,np.pi,n_out))
		inner_circ_x = 2 - 2 * np.cos(np.linspace(0, np.pi,n_in)) 
		inner_circ_y = 2 - 2 * np.sin(np.linspace(0, np.pi,n_in)) - 1.1 

		X = np.vstack([np.append(outer_circ_x,inner_circ_x), 
						np.append(outer_circ_y, inner_circ_y)]).T
		y = np.hstack([np.zeros(n_out, dtype=np.intp), 
						np.ones(n_in, dtype=np.intp)])

		X,y = shuffle(X,y, random_state=10)
		
		noise_a = np.random.normal(0,0.1,self.n)
		noise_b = np.random.normal(0,0.1,self.n)

		X += np.array([noise_a, noise_b]).T

		return X,y

if __name__ == '__main__':
	w = 5
	r = 5
	d = 1
	n = 1000

	dm = DoubleMoon(w,r,d,n)

	X,y = dm.get_values()

	print(X.shape)
	plt.scatter(X[:,0],X[:,1])
	plt.show()


