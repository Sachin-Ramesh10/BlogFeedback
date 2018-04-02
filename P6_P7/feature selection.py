from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


train = pd.read_csv('blogData_train.csv', sep=',', header=None)
test = pd.read_csv('blogData_test-2012.03.31.01_00.csv', sep=',', header=None)
x_train = train.iloc[:, :-1]
x_test = test.iloc[:, :-1]
traintar = train.iloc[:, -1]
testtar = test.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(x_train)

names = []
for i in range(len(x_train.columns)):
    names.append(i)

lasso = LassoCV()
lasso.fit(X, traintar)


def print_linear(coefs, names, sort):
    # if names == None:
    names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                      for coef, name in lst)


print "Lasso model: ", print_linear(lasso.coef_, names, sort=True)