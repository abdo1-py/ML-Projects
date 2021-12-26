#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
#----------------------------------------------------

#load breast cancer data

path='F:\\Data science\\Data projects\\Classfication\\Data1.txt'
data=pd.read_csv(path,header=3,names=['effporosity','Vshale','Class'] )
#X Data

X = data.iloc[:,:-1] 
y = data.iloc[:, -1]

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying BernoulliNB Model 

'''
#sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True,class_prior=None)
'''

BernoulliNBModel = BernoulliNB(alpha=0.1,binarize=1, fit_prior=True)
BernoulliNBModel.fit(X_train, y_train)

#Calculating Details
print('BernoulliNBModel Train Score is : ' , BernoulliNBModel.score(X_train, y_train))
print('BernoulliNBModel Test Score is : ' , BernoulliNBModel.score(X_test, y_test))
# print('----------------------------------------------------')
# #Calculating Prediction
y_pred = BernoulliNBModel.predict(X_test)
y_pred_prob = BernoulliNBModel.predict_proba(X_test)
# print('Predicted Value for BernoulliNBModel is : ' , y_pred[:10])
# print('Prediction Probabilities Value for BernoulliNBModel is : ' , y_pred_prob[:10])


# #----------------------------------------------------
# Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
 
