import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'F:\\Data science\\Data projects\\Reg for more two varaible\\Data Well .txt'
data = pd.read_csv(path, header=None, names=['Depth', 'Vshale', 'effporosity','Density'])

# show data
# print('data = ')
# print(data.head(10) )
# print()
# print('data.describe = ')
# print(data.describe())

# rescaling data
data = (data - data.mean()) / data.std()

# print()
# print('data after normalization = ')
# print(data.head(10) )


# add ones column
data.insert(0, 'Ones', 1)


# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


# print('**************************************')
# print('X2 data = \n' ,X.head(10) )
# print('y2 data = \n' ,y.head(10) )
# print('**************************************')


# convert to matrices and initialize theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0,0]))


# print('X2 \n',X)
# print('X2.shape = ' , X.shape)
# print('**************************************')
# print('theta2 \n',theta)
# print('theta2.shape = ' , theta.shape)
# print('**************************************')
# print('y2 \n',y)
# print('y2.shape = ' , y.shape)
# print('**************************************')


alpha = 0.1
iters = 100



def computeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2)
    print('z \n',z)
    print('m ' ,len(X))
    return np.sum(z) / (2 * len(X))

# print('computeCost(X, y, theta) = ' , computeCost(X, y, theta))

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
g,cost= gradientDescent(X, y, theta, alpha, iters)

# print('g = ' , g)
# print('cost  = ' , cost[0:50] )
# print('computeCost = ' , computeCost(X, y, g))
# print('**************************************')



# initialize variables for learning rate and iterations

# perform linear regression on the data set
g, cost= gradientDescent(X, y, theta, alpha, iters)

# get the cost (error) of the model
thiscost = computeCost(X, y, g)


# print('g = ' , g)
# print('cost  = ' , cost[0:50] )
# print('computeCost = ' , thiscost)
# print('**************************************')




x = np.linspace(data.Vshale.min(), data.Vshale.max(), 100)
# # print('x \n',x)
# # print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
# print('f \n',f)

# draw the line for effporosity vs. Vshale

# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Vshale, data.effporosity, label='Training Data')
# ax.legend(loc=2)
# ax.set_xlabel('Vshale')
# ax.set_ylabel('effporosity')
# ax.set_title('Vshale vs. effporosity')


# get best fit line for effporosity vs. Density

# x = np.linspace(data.effporosity.min(), data.effporosity.max(), 100)
# # print('x \n',x)
# # print('g \n',g)

# f = g[0, 0] + (g[0, 1] * x)
# # # print('f \n',f)

# # draw the line  for effporosity vs. Density

# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.effporosity, data.Density, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('effporosity')
# ax.set_ylabel('Density')
# ax.set_title('effporosity vs. Density')


# get best fit line for effporosity vs. Density

x = np.linspace(data.Vshale.min(), data.Vshale.max(), 100)
# print('x \n',x)
# print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
# # print('f \n',f)

# draw the line  for effporosity vs. Density

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Vshale, data.Density, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Vshale')
ax.set_ylabel('Density')
ax.set_title('Vshale vs. Density')




# # draw error graph

# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')


