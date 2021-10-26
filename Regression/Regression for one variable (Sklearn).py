#Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

#----------------------------------------------------

#load The data

path='F:\\Data science\\Data projects\\Reg for one variable\\Well Data.txt'
d=pd.read_csv(path,header=None,names=['effporosity','Vshale'])
X = d.iloc[:,:1]
y = d.iloc[:, -1]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Simple Linear Regression to the Training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

print('The Reg Score train and test',regressor.score(X_train, y_train),' , ',regressor.score(X_test, y_test))

# Predicting the Test set results
y_pred = regressor.predict(X_test)
 
# print(y_pred[:5])
# print(y_test[:5])
##################################################################

#applying Ridge Regression

RidgeRegressionModel = Ridge(alpha=1.0,random_state=33)
RidgeRegressionModel.fit(X_train, y_train)
Ridge_train_score = RidgeRegressionModel.score(X_train,y_train)
Ridge_test_score = RidgeRegressionModel.score(X_test, y_test)

print('The Ridge Score train is ',Ridge_train_score,'The Ridge Score test is ',Ridge_test_score)
############################################################################
#Applying Lasso Regression Model
 
LassoRegressionModel = Lasso(alpha=1.0,random_state=33,normalize=False)
LassoRegressionModel.fit(X_train, y_train)
print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))
print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))

#######################################################################
#Applying SGDRegressor Model 
# SGDRegressionModel = SGDRegressor(alpha=0.01,random_state=33,penalty='l2',loss = 'huber')
# SGDRegressionModel.fit(X_train, y_train)

# print('SGD Regression Train Score is : ' , SGDRegressionModel.score(X_train, y_train))
# print('SGD Regression Test Score is : ' , SGDRegressionModel.score(X_test, y_test))


##########################################

# print(mean_absolute_error(y_test, y_pred))

# print(mean_squared_error(y_test, y_pred))

# print(median_absolute_error(y_test, y_pred))


##################################################
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel('Effectiveporosity')
plt.ylabel('Vshale')
plt.show()
 

