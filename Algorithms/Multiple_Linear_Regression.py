import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("../data/50_Startups.csv")
#Feature set
X = dataset.iloc[:,:-1].values
#Dependable variable
y = dataset.iloc[:,-1].values

#Encoding the categorizable variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder() ,[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Spliting the dataset to train and test 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

#Class is taking care of dummy variable trap and performs bakcward elimination to find the biggest p value
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

# print(regressor.predict)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

print(regressor.predict([[1,0,0,160000,130000,300000]]))
print(regressor.coef_)
print(regressor.intercept_)