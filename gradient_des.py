import numpy as np 

data = np.loadtxt('one_var.txt',delimiter=',',dtype=np.float)
y = data[:,-1]
y = y.reshape((-1,1))
x = np.ones((y.shape[0],1))
indices = np.arange(data.shape[1])
a =  data[:,indices!=indices.shape[0]-1]
a = a.reshape((y.shape[0],data.shape[1]-1))
x = np.append(x,a,axis=1)


# Mean Normalization

# print x.shape
# for i in range (1,x.shape[1]):
# 	for j in range (0,x.shape[0]):
# 		x[j,i] = (x[j,i]-x[:,i].mean())/(x[:,i].max()-x[:,i].min())
# print x
# print x.shape

class Gradient_descent:
	def __init__(self, x,y,alpha):
		self.x = x
		self.y = y
		self.alpha = alpha
		self.m = x.shape[0]
		self.theta = np.zeros((x.shape[1],1))
	def loss_function(self,t):
		return (np.dot(self.x,t) - self.y)  # if i change here then sign in equation will change
	def run_Gd(self,epochs):
		for i in range(0,epochs):
			gradient = np.dot(self.x.transpose(),self.loss_function(self.theta))
			print((self.loss_function(self.theta)**2).sum()/(2*self.m))
			self.theta = self.theta - gradient*self.alpha/self.m



gd = Gradient_descent(x,y,1)
gd.run_Gd(10000)
print gd.theta


