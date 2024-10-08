#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
#----------------------------------------------------

path = 'F:\\Data science\\Data projects\\Reg for more two varaible\\Well Data Re.txt'
data = pd.read_csv(path, header=None, names=["Density","Gamma",	"Vshale","Effeporo"])



cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying Random Forest Regressor Model 

'''
sklearn.ensemble.RandomForestRegressor(n_estimators='warn', criterion=’mse’, max_depth=None,
                                       min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                       max_features=’auto’, max_leaf_nodes=None,min_impurity_decrease=0.0,
                                       min_impurity_split=None, bootstrap=True,oob_score=False, n_jobs=None,
                                       random_state=None, verbose=0,warm_start=False)
'''

RandomForestRegressorModel = RandomForestRegressor(n_estimators=10,max_depth=2, random_state=33)
RandomForestRegressorModel.fit(X_train, y_train)

# #Calculating Details
# print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
# print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))
# print('Random Forest Regressor No. of features are : ' , RandomForestRegressorModel.n_features_)
# # print('----------------------------------------------------')


GBRModel = GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate = 1.5 ,random_state=33)
GBRModel.fit(X_train, y_train)

#Calculating Details
print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = GBRModel.predict(X_test)
print('Predicted Value for GBRModel is : ' , y_pred[:10])

#--------------------------------------------------


# #Calculating Prediction
# y_pred = RandomForestRegressorModel.predict(X_test)
# print('Predicted Value for Random Forest Regressor is : ' , y_pred[:10])

# #----------------------------------------------------
# #Calculating Mean Absolute Error
# MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Absolute Error Value is : ', MAEValue)

# #----------------------------------------------------
# #Calculating Mean Squared Error
# MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Squared Error Value is : ', MSEValue)

# #----------------------------------------------------
# #Calculating Median Squared Error
# MdSEValue = median_absolute_error(y_test, y_pred)
# print('Median Squared Error Value is : ', MdSEValue )
 





