
#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing the datasets and dropping the instance coulumn
dataset=pd.read_csv('C:/Users/BadBoy/Desktop/kaggle/data.csv')
dataset=dataset.drop('Instance',axis=1)
testset=pd.read_csv('C:/Users/BadBoy/Desktop/kaggle/testing.csv')
testset=testset.drop('Instance',axis=1)

# merging the train and test dataset and checking for nan values
merge=pd.concat([dataset,testset])
null_columns=merge.columns[merge.isnull().any()]

# removing the nan values
merge['Gender'] = merge['Gender'].fillna(method='ffill')
merge['Hair Color'] = merge['Hair Color'].fillna(method='ffill')
merge['University Degree'] = merge['University Degree'].fillna(method='ffill')
merge['Profession'] = merge['Profession'].fillna(method='ffill')
merge['Income'] = merge['Income'].fillna(method='ffill')
merge['Year of Record']=merge['Year of Record'].fillna((merge['Year of Record'].mean()))
merge['Age']=merge['Age'].fillna((merge['Age'].mean()))
merge['Income in EUR']=merge['Income in EUR'].fillna((merge['Income in EUR'].mean()))

# checking if nan values are left
p=merge[null_columns].isnull().sum()
print(p)

# categorizing the labeleb values
merge = pd.get_dummies(merge,columns=['Country'])
merge= pd.get_dummies(merge,columns=['Gender'])
merge = pd.get_dummies(merge,columns=['Profession'])
merge= pd.get_dummies(merge,columns=['University Degree'])
merge= pd.get_dummies(merge,columns=['Hair Color'])

# splitting the merged dataset into training and testing dataset
data=merge.head(111993)
data=data.drop('Income',axis=1)
p=data[null_columns].isnull().sum()
print(p)
test=merge.head(73230)
test=test.drop('Income in EUR',axis=1)
test=test.drop('Income',axis=1)

# removing the outliers from age
sns.boxplot(x=dataset["Age"])
data=data[data.Age<80]
dataset=data[data.Age>20]

# removing the outliers from height column
data = data.rename(columns={'Body Height [cm]': 'Height'})
sns.boxplot(x=data["Height"])
data=data[data.Height>120]
data=data[data.Height<220]
sns.boxplot(x=data["Height"])

#removing the outliers from income column

data = data.rename(columns={'Income in EUR': 'Income'})
sns.boxplot(x=data["Income"])
data=data[data.Income<150000]
sns.boxplot(x=data["Income"])

#splitting the train set in dependent and independent variables

Y = data.Income
X = data.drop("Income", axis=1)
Y = pd.DataFrame(Y)
Y.head()

#splitting the dataset into testing and traing set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# applying linear regression

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
Ypred = lm.predict(X_test)
# calculating hte rmse
mse = np.square(np.subtract(Y_test,Ypred)).mean()
import math
rmse = math.sqrt(mse)
print(rmse)

#predicting for the test set
prediction= lm.predict(test)

# exporting the result
np.savetxt('C:/Users/BadBoy/Desktop/kaggle/save.csv',prediction, delimiter=',')

