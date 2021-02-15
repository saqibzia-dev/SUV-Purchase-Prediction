import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]]
Y = dataset.iloc[:,4]
"""handling missing values"""
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values = np.nan,strategy="mean")
# X = imputer.fit_transform(X)
"""handling categorical data"""
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# X[:,0] = LabelEncoder().fit_transform(X[:,0])
# X[:,1] = LabelEncoder().fit_transform(X[:,1])
# from sklearn.compose import ColumnTransformer
# col_transformer = ColumnTransformer([("encoding",OneHotEncoder(),[0,1])],
#                                     remainder="passthrough")
# X = (col_transformer.fit_transform(X)).toarray()
"""splitting the data"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)
"""Standardizing our data""" #don't require in RF only doing 
#only doing to speed up plotting the decision boundry process
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
"""modeling our data"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="gini")
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
"""Assessing model performance"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
"""Plotting decision boundry on training data"""
from matplotlib.colors import ListedColormap
XX,YY = x_train,y_train
X1 = np.arange(start = min(XX[:,0])-1,stop = max(XX[:,0])+1,step=0.01)
X2 = np.arange(start = min(XX[:,1])-1,stop = max(XX[:,1])+1,step=0.01)
X1,X2 = np.meshgrid(X1,X2)
Z = np.array([X1.ravel(),X2.ravel()]).T
plt.contourf(X1,X2,classifier.predict(Z).reshape(X1.shape),
             alpha = 0.75,cmap =ListedColormap(("red","green")) )
plt.scatter(XX[:,0],XX[:,1],c=YY)
plt.xlim(min(XX[:,0])-1,max(XX[:,0])+1)
plt.ylim(min(XX[:,1])-1,max(XX[:,1])+1)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Random Forest Classifier on Training data")
plt.figure("Test Data",clear=True)
from matplotlib.colors import ListedColormap
XX,YY = x_test,y_test
X1 = np.arange(start = min(XX[:,0])-1,stop = max(XX[:,0])+1,step=0.01)
X2 = np.arange(start = min(XX[:,1])-1,stop = max(XX[:,1])+1,step=0.01)
X1,X2 = np.meshgrid(X1,X2)
Z = np.array([X1.ravel(),X2.ravel()]).T
plt.contourf(X1,X2,classifier.predict(Z).reshape(X1.shape),
             alpha = 0.75,cmap =ListedColormap(("red","green")) )
plt.scatter(XX[:,0],XX[:,1],c=YY)
plt.xlim(min(XX[:,0])-1,max(XX[:,0])+1)
plt.ylim(min(XX[:,1])-1,max(XX[:,1])+1)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Random Forest Classifier on Test data")
plt.show()


