# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 02:14:27 2019

@author: JohitGarg
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset['Stay_In_Current_City_Years'] = dataset['Stay_In_Current_City_Years'].replace(to_replace="4+", value=4)
dataset['Product_Category_2'] = dataset['Product_Category_2'].fillna(0)
dataset['Product_Category_3'] = dataset['Product_Category_3'].fillna(0)
X = dataset.iloc[:, 0:].values
y = dataset.iloc[:, 11].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_shape = y_train.shape
y_train = sc_y.fit_transform(np.reshape(a=y_train,newshape = (-1,1) ))
y_train = np.reshape(y_train,y_shape)"""

# Fitting the Regression Model to the dataset
from xgboost import XGBRegressor
regressor = XGBRegressor(learning_rate = 0.1, gamma = 0.1)
regressor.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
parameters = [{'learning_rate': [0.01, 0.1, 0.2], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Predicting new values with test data
y_pred = regressor.predict(X_test)

# Finding the rmse value
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
rmse = math.sqrt(mean_squared_error(y_test,y_pred))