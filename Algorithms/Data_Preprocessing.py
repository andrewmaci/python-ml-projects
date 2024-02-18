import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv("../data/Data_Preprocessing_Data.csv")
#Feature set
X = dataset.iloc[:,:-1].values
#Dependable variable
y = dataset.iloc[:,-1].values

Imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
Imputer.fit(X[:,1:3])
X[:,1:3] = Imputer.transform(X[:,1:3])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder() ,[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

le = LabelEncoder();
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

print(X_train,X_test)