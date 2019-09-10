import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score

x = np.random.rand(100,1)
y = 5*x*x + 0.1*np.random.randn(100,1)

X = np.zeros((100,2))

X[:,0] = 1

# can this be done without a for-loop?
for i in range(100):
    X[i,1] = x[i]

# manual inversion
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# scikit-learn
clf = skl.LinearRegression().fit(X,y)
y_tilde = clf.predict(X)

# MSE
MSE = mean_squared_error(y,y_tilde)

# R²
R_squared = r2_score(y,y_tilde)

print("MSE: %.2f" %MSE)
print("R_squared: %.2f" %R_squared)


# own MSE function
def MSE(y, y_tilde):
    n = len(y)
    MSE = 1/float(n) * np.sum((y-y_tilde)**2)
    return MSE

# own R_squared function
def R_squared(y, y_tilde):
    n = len(y)
    y_bar = 1/float(n) * np.sum(y)
    R_squared = 1 - (np.sum((y-y_tilde)**2)/np.sum((y-y_bar)**2))
    return R_squared
