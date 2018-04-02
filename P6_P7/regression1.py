import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

train = pd.read_csv('blogData_train.csv', sep = ',', header = None)
test = pd.read_csv('blogData_test-2012.03.31.01_00.csv', sep = ',', header = None)
x_train = train.iloc[:, :-1]
x_test = test.iloc[:, :-1]

traintar = train.iloc[:,-1]
testtar = test.iloc[:,-1]

linreg = LinearRegression()
from sklearn.grid_search import GridSearchCV

parameters =  {'fit_intercept':[True,False], 'normalize':[True, False], 'copy_X': [True, False], 'n_jobs': [1,2,3]}

grid_obj = GridSearchCV(linreg,parameters)
grid_fit = grid_obj.fit(x_train,traintar)
best_clf = grid_fit.best_estimator_
best_predictions = best_clf.predict(x_test)

er = np.sqrt(np.mean((best_predictions - testtar)**2))
print er

from sklearn.linear_model import RidgeCV

ridreg = RidgeCV(alphas=(0.1, 1.0, 10.0), gcv_mode= 'auto')
s = ridreg.fit(x_train,traintar)
predict = s.predict(x_test)

er = np.sqrt(np.mean((predict - testtar)**2))
print er

from sklearn.linear_model import LassoCV

parameters = {'eps' :[0.0001,0.00001,0.0000001],
                'n_alphas': [100,250,300],
                'tol': [0.1, 0.001,0.0001],
                'selection' :['cyclic', 'random']}

lasreg = LassoCV()

grid_obj = GridSearchCV(lasreg,parameters)
grid_fit = grid_obj.fit(x_train,traintar)
best_clf = grid_fit.best_estimator_
bestlas = best_clf.predict(x_test)

er = np.sqrt(np.mean((bestlas - testtar)**2))
print er