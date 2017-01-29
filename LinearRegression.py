import numpy as np


def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


class LinearRegression:
	def __init__(self,alpha=0.01,iters=1000):
		'''
		a class to fit, predict and score a lin reg model 
			w pure python... at some point ugh lol
		'''
		self.theta = []
		self.alpha = alpha
		self.iters = iters

	def normalize(self,X,y):
		
		data = np.concatenate([X.T,y]).T
		mean = data.mean()
		std = data.std()  
		
		# Normalize
		data = (data - mean) / std

		n_cols = data.shape[1]

		a = np.ones(len(X))
		a = np.matrix(a)
		data = np.concatenate([a,data.T]).T

		self.X = data[:,:n_cols]  
		self.y = data[:,n_cols:]

		return self.X,self.y,mean,std

	def validate(self,x):
		'''
		make sure input is in proper form (matrix)
		'''
		return np.matrix(x)

	def fit(self, X,y):

		X = self.validate(X)
		y = self.validate(y)

		self.X,self.y,self.mean,self.std = self.normalize(X,y)

		n_cols = self.X.shape[1]
		self.theta = np.matrix(np.zeros(n_cols))
		self.theta, self.cost = gradientDescent(self.X, 
								self.y, 
								self.theta, 
								self.alpha, 
								self.iters)
		return self

	def score(self, X_test,y_test):
		X_test = self.validate(X_test)
		y_test = self.validate(y_test)

		X_test,y_test,_,_ = self.normalize(X_test,y_test)
		return 1 - computeCost(X_test,y_test, self.theta)

	def predict(self, X_test):
		X_test = self.validate(X_test)

		dummy_y = np.matrix(np.zeros(X_test.shape[0]))
		X_test,y_test,_,_ = self.normalize(X_test,dummy_y)
		return (X_test * self.theta.T * self.std) + self.mean

def test_lin_reg():
	# convert to matrices and initialize theta
	data = [[2104,3,399900],
		[1600,3,329900],
		[2400,3,369000],
		[1416,2,232000],
		[3000,4,539900]]
	data = np.matrix(data)  

	data = (data - data.mean()) / data.std()  

	a = np.ones(5)
	a = np.matrix([a])
	data = np.concatenate([a,data.T]).T
	X2 = data[:,:3]  
	y2 = data[:,3:]  
	print(X2,y2)
	theta2 = np.matrix(np.array([0,0,0]))
	print(theta2)

	# perform linear regression on the data set
	alpha = 0.01  
	iters = 1000
	g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

	# get the cost (error) of the model
	cost = computeCost(X2, y2, g2)  
	print(cost)

if __name__ == '__main__':
	data = [[2104,3,399900],
		[1600,3,329900],
		[2400,3,369000],
		[1416,2,232000],
		[3000,4,539900]]
	data = np.matrix(data)  
	X = data[:,:2]
	y = data[:,2:].T

	model = LinearRegression()
	model.fit(X,y)
	print(model.score(X,y))
	print(model.predict(X))

