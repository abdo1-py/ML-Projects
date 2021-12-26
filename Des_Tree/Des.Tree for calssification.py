#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
import pandas as pd
#----------------------------------------------------

path='F:\\Data science\\Data projects\\Classfication\\Data1.txt'
d=pd.read_csv(path,header=3,names=['effporosity','Vshale','Class'])
X = d.iloc[:,:-1] 
y = d.iloc[:, -1]
#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#Applying DecisionTreeRegressor Model 

'''
sklearn.tree.DecisionTreeRegressor(criterion='mse’, splitter=’best’, max_depth=None,min_samples_split=2,
                                   min_samples_leaf=1,min_weight_fraction_leaf=0.0, max_features=None,
                                   random_state=None, max_leaf_nodes=None,min_impurity_decrease=0.0,
                                   min_impurity_split=None, presort=False)
'''

DecisionTreeRegressorModel = DecisionTreeRegressor( max_depth=2,random_state=33)
DecisionTreeRegressorModel.fit(X_train, y_train)

#Calculating Details
print('DecisionTreeRegressor Train Score is : ' , DecisionTreeRegressorModel.score(X_train, y_train))
print('DecisionTreeRegressor Test Score is : ' , DecisionTreeRegressorModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = DecisionTreeRegressorModel.predict(X_test)
print('Predicted Value for DecisionTreeRegressorModel is : ' , y_pred[:10])
y_pred[:10]
print(list(y_test[:10]))

#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )
 
