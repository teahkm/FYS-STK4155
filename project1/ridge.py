import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from common import franke_function, x, y, CreateDesignMatrix_X

#try with noise
z = franke_function(x,y) #+ np.random.randn(20,20)
z_flat = np.ravel(z)


x_flat = np.ravel(x)
y_flat = np.ravel(y)

X = CreateDesignMatrix_X(x,y,n=2)


# The Ridge regression with a hyperparameter lambda = 0.1
_lambda = 0.1
#clf_ridge = skl.Ridge(alpha=_lambda).fit(X, z_flat)
#z_ridge = clf_ridge.predict(X)

def Ridge(X, y, _lambda=0.1):
    XtX = X.T.dot(X) #help variable
    beta = np.linalg.inv(XtX + _lambda*np.identity(len(XtX[0]))).dot(X.T).dot(y)
    ytilde = X.dot(beta)

    return beta, ytilde

beta, z_ridge = Ridge(X, z_flat, 0.1)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(z_flat, z_ridge))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(z_flat, z_ridge))
# Mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(z_flat, z_ridge))
#print(clf_ridge.coef_, clf_ridge.intercept_)
