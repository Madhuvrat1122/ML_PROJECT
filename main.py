import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
features_train = pd.read_csv("Dataset/train.csv").drop(['Activity','subject'],axis=1)
labels_train=pd.read_csv("Dataset/train.csv")['Activity']
features_test=pd.read_csv("Dataset/test.csv").iloc[:,:-2]
labels_test=pd.read_csv("Dataset/test.csv").iloc[:,-1]
from sklearn import preprocessing  
label_encoder = preprocessing.LabelEncoder()  
labels_train= label_encoder.fit_transform(labels_train) 
labels_test= label_encoder.transform(labels_test)
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test) 
from sklearn.metrics import accuracy_score
Decision_acc=(accuracy_score(labels_test, labels_pred)*100)
print("Accuracy score is:- ",accuracy_score(labels_test, labels_pred))
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = labels_train, exog = features_train).fit()
list1=list(regressor_OLS.pvalues)
item=[]
for i in list1:
    if i<0.05:
        item.append(i)
features_train = pd.read_csv("Dataset/train.csv").iloc[:,item]
features_test = pd.read_csv("Dataset/test.csv").iloc[:,item]
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test) 
from sklearn.metrics import accuracy_score
Decision_fet_acc=(accuracy_score(labels_test, labels_pred)*100)
print("Accuracy score After Features Selection is:- ",accuracy_score(labels_test, labels_pred))

###random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test)
from sklearn.metrics import accuracy_score
random_acc=(accuracy_score(labels_test, labels_pred)*100)
print("Accuracy score is:- ",accuracy_score(labels_test, labels_pred))

##after features selection
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = labels_train, exog = features_train).fit()
list1=list(regressor_OLS.pvalues)
item=[]
for i in list1:
    if i<0.05:
        item.append(i)
features_train = pd.read_csv("Dataset/train.csv").iloc[:,item]
features_test = pd.read_csv("Dataset/test.csv").iloc[:,item]
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train) 
from sklearn.metrics import accuracy_score
random_fet_acc=(accuracy_score(labels_test, labels_pred)*100)
print("Accuracy score After Features selection is:- ",accuracy_score(labels_test, labels_pred))

###kNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) 
classifier.fit(features_train, labels_train)
probability = classifier.predict_proba(features_test)
labels_pred = classifier.predict(features_test)
from sklearn.metrics import accuracy_score
knn_acc=(accuracy_score(labels_test, labels_pred)*100)
print("Accuracy score is:- ",accuracy_score(labels_test, labels_pred))
