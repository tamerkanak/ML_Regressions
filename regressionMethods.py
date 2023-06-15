# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 02:02:41 2023

@author: tamer
"""


#1.libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm


#data loading
data = pd.read_csv('maaslar_yeni.csv')


#slicing
x = data.iloc[:,2:5]
y = data.iloc[:,-1:]


#numpy array transforming
X = x.values
Y = y.values





#linear regression 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)


print("Multiple Linear R2 Value:")
print(r2_score(Y,lin_reg.predict(X)))


model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())





#polynomial regression 
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

print("Polynomial R2 Value:")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print("Poly OLS")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())





#scaling of data
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))





#Support Vector Regression
from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("SVR R2 Value:")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())





#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print("Decision Tree R2 Value:")
print(r2_score(Y,r_dt.predict(X)))

print("DT OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())





#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print("Random Forest R2 Value:")
print(r2_score(Y,rf_reg.predict(X)))

print("RF OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
