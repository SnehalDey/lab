# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:24:30 2019

@author: BadBoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset=pd.read_csv('C:/Users/BadBoy/Desktop/kaggle/data.csv')
dataset=dataset.drop('Instance',axis=1)
testset=pd.read_csv('C:/Users/BadBoy/Desktop/kaggle/testing.csv')
testset=testset.drop('Instance',axis=1)
merge=pd.concat([dataset,testset])
null_columns=merge.columns[merge.isnull().any()]


merge['Gender'] = merge['Gender'].fillna(method='ffill')
merge['Hair Color'] = merge['Hair Color'].fillna(method='ffill')
merge['University Degree'] = merge['University Degree'].fillna(method='ffill')
merge['Profession'] = merge['Profession'].fillna(method='ffill')
merge['Income'] = merge['Income'].fillna(method='ffill')
merge['Year of Record']=merge['Year of Record'].fillna((merge['Year of Record'].mean()))
merge['Age']=merge['Age'].fillna((merge['Age'].mean()))
merge['Income in EUR']=merge['Income in EUR'].fillna((merge['Income in EUR'].mean()))

p=merge[null_columns].isnull().sum()
print(p)

merge = pd.get_dummies(merge,columns=['Country'])
merge= pd.get_dummies(merge,columns=['Gender'])
merge = pd.get_dummies(merge,columns=['Profession'])
merge= pd.get_dummies(merge,columns=['University Degree'])
merge= pd.get_dummies(merge,columns=['Hair Color'])
data=merge.head(111993)
data=data.drop('Income',axis=1)
###p=data[null_columns].isnull().sum()
###print(p)
test=merge.head(73230)
test=test.drop('Income in EUR',axis=1)
test=test.drop('Income',axis=1)


data = data.rename(columns={'Body Height [cm]': 'Height'})
###
data=data[data.Height>120]
data=data[data.Height<220]
sns.boxplot(x=data["Height"])
#print(test)
#

data = data.rename(columns={'Income in EUR': 'Income'})
data=data[data.Income<150000]
sns.boxplot(x=data["Income"])
##
##data,testdata=(merge,test_size=0.3953612672292318)
##
Y = data.Income
X = data.drop("Income", axis=1)
Y = pd.DataFrame(Y)
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
##
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
Ypred = lm.predict(X_test)
mse = np.square(np.subtract(Y_test,Ypred)).mean()
import math
rmse = math.sqrt(mse)
print(rmse)
prediction= lm.predict(test)
np.savetxt('C:/Users/BadBoy/Desktop/kaggle/save.csv',prediction, delimiter=',')

