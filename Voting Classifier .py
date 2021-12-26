# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

#----------------------------------------------------
#Applying VotingClassifier Model 

'''
#ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)
'''

#loading models for Voting Classifier
DTModel_ = DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state = 33)
LDAModel_ = LinearDiscriminantAnalysis(n_components=1 ,solver='svd')
SGDModel_ = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)

#loading Voting Classifier
VotingClassifierModel = VotingClassifier(estimators=[('DTModel',DTModel_),('LDAModel',LDAModel_),('SGDModel',SGDModel_)], voting='hard')
VotingClassifierModel.fit(X_train, y_train)

#Calculating Details
print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))
print('----------------------------------------------------')

# #Calculating Prediction
# y_pred = VotingClassifierModel.predict(X_test)
# print('Predicted Value for VotingClassifierModel is : ' , y_pred[:10])

# #----------------------------------------------------
# #Calculating Confusion Matrix
# CM = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix is : \n', CM)

# # drawing confusion matrix
# sns.heatmap(CM, center = True)
# plt.show()
 
