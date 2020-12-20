# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:52:18 2020

@author: Husam
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
# it was supposed to call the dirname, but it did not work so I downloaded file
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

cars = pd.read_csv('./USA_cars_datasets.csv')

##checking data
cars.head()

cars.drop(['vin', 'lot', 'Unnamed: 0', 'condition', 'country'], axis=1, inplace=True)
cars.head()
cars.info()

cars.describe().style.format('{:0.2f}')
cars.isnull().sum()


# Plotting of Price against Milage/Year
sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))

ax1.scatter(x='mileage', y='price', data=cars, marker='.', linewidth= .3, s=140)

ax2.scatter(x='year', y='price', data=cars, marker='.', linewidth= .3, s=140)

ax1.set_title('Price vs Mileage', fontsize=20, fontweight='bold')
ax1.set_ylabel('Price',fontsize=20)
ax1.set_xlabel('Mileage',fontsize=20)


ax2.set_title('Price vs Year', fontsize=20, fontweight='bold')
ax2.set_ylabel('Price',fontsize=20)

ax2.set_xlabel('Year',fontsize=20)

# Plotting of proportion of the vehicles title status
sns.scatterplot(y='price', x='mileage', data=cars, hue='title_status')

# Plotting the distributions of Prices 
plt.figure(figsize=(10, 10))
price_distribution = plt.hist(x='price', data=cars, bins=30, alpha=.7)
plt.title('Price Distribution', fontsize=20, fontweight='bold')
plt.xlabel('Price', fontsize=20)
plt.ylabel('Quantity', fontsize=20)


brand = pd.get_dummies(cars['brand'], drop_first=True)
model= pd.get_dummies(cars['model'], drop_first=True)
title = pd.get_dummies(cars['title_status'], drop_first=True)
color = pd.get_dummies(cars['color'], drop_first=True)
state = pd.get_dummies(cars['state'], drop_first=True)

cars_modelling_data = pd.concat([cars, brand, model, title, state, color], axis=1)

cars_modelling_data.drop(['brand', 'model', 'state','title_status', 'mileage', 'color'], axis=1,inplace=True)

X = cars_modelling_data.drop(['price'], axis=1)
y = cars_modelling_data['price']
print(X.shape, y.shape)



from sklearn.model_selection import KFold

kfold = KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#initialise params to prove
params = {'fit_intercept':[True, False], 'normalize':[True, False], 'copy_X': [True, False], 'n_jobs': [1, 2, 3, 4, 5, 10, 15]}

linear_reg = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
clf = GridSearchCV(linear_reg, param_grid=params)
clf.fit(X, y)
#print(clf.score(X, y))
#print(clf.predict(X))
#print(clf.best_params_)
#print(clf.best_score_)

linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
#print(linear_reg.score(X, y))
coef = linear_reg.coef_

#Check how well it is performing
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('MAE', np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))

plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred,marker='.',c='red',edgecolors='black',s=100,linewidth=1)

plt.figure(figsize=(10, 10))
plt.hist((y_test - y_pred), bins=30, color='r')


###############################################################################


# Logistic Regression and Random Forest Classifier
#kfold = KFold(n_splits=2)
X = cars_modelling_data.drop(['salvage insurance'], axis=1)
y = cars_modelling_data['salvage insurance']
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
kfold = KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
logic_reg = LogisticRegression(solver='liblinear', penalty='l1')

#grid_values = {'solver': ['liblinear'], 'max_iter': [100], 'penalty': ['l1', 'l2'],'C':np.logspace(-4, 4, 20)}
#clf = GridSearchCV(logic_reg, param_grid = grid_values,scoring = 'recall')
#clf.fit(X_train, y_train)

print(clf.best_params_)
print(clf.best_score_)

best_clf = clf.fit(X_train, y_train)
logic_reg.fit(X_train, y_train)
#print(logic_reg.)

from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
y_pred = clf.predict(X_test)

disp = plot_confusion_matrix(best_clf, X_test, y_test, cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')

print(title)
print(disp.confusion_matrix)

accuracy = (disp.confusion_matrix[0][0] + disp.confusion_matrix[1][1]) / (disp.confusion_matrix[0][0] + disp.confusion_matrix[0][1] + disp.confusion_matrix[1][0] + disp.confusion_matrix[1][1])
precision = (disp.confusion_matrix[0][0]) / (disp.confusion_matrix[0][0] + disp.confusion_matrix[0][1])
recall = (disp.confusion_matrix[0][0]) / (disp.confusion_matrix[0][0] + disp.confusion_matrix[0][1])
F1 = 2 * precision * recall / (precision + recall)
print('accuracy', accuracy)
print('precision', precision)
print('recall', recall)
print('F1', F1)














