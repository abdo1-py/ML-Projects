import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np




# Importing the dataset
path = 'F:\\Data science\\Data projects\\Reg for more two varaible\\Well Data Re.txt'
data = pd.read_csv(path, header=None, names=["Density","Gamma",	"Vshale","Effeporo"])

cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(data)
data = imp.transform(data)






# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# print(regressor.score(X_train, y_train),regressor.score(X_test, y_test))

# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# print(y_pred)
# print(y_test)

# #####################################################################
 
# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(y_test, y_pred))

# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_pred))

# from sklearn.metrics import median_absolute_error
# print(median_absolute_error(y_test, y_pred))


# # Visualising the Training set results

                   #Effeporo $    Vshale
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(X_train['Vshale'],y_train['Effeporo'],c='g',marker='o',label='Vshale')
ax.scatter(X_train['Vshale'],y_train['Effeporo'],c='y',marker='x',label='Effeporo')
ax.set_xlabel('Vshale')
ax.set_xlabel('effporosity')
ax.legend()
plt.show()

                    # Density $    Vshale
                      
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(X_train['Density'],y_train['Effeporo'],c='b',marker='o',label='Density')
ax.scatter(X_train['Density'],y_train['Effeporo'],c='r',marker='x',label='Effeporo')
ax.set_xlabel('Density')
ax.set_xlabel('effporosity')
ax.legend()
plt.show()
                     # Gamma $    Vshale
                      
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(X_train['Gamma'],y_train['Effeporo'],c='g',marker='o',label='Density')
ax.scatter(X_train['Gamma'],y_train['Effeporo'],c='r',marker='x',label='Effeporo')
ax.set_xlabel('Gamma')
ax.set_xlabel('effporosity')
ax.legend()
plt.show()




