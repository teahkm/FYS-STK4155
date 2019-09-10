import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score

#make data
x = np.arange(0,1,0.05)
y = np.arange(0,1,0.05)
x,y = np.meshgrid(x,y)

#franke function
def franke_function(x,y):
    term1 = 3/4*np.exp(-(9*x-2)**2/4 - (9*y-2)**2/4)
    term2 = 3/4*np.exp(-(9*x+1)**2/49 - (9*y+1)/10)
    term3 = 1/2*np.exp(-(9*x-7)**2/4-(9*y-3)**2/4)
    term4 = -1/5*np.exp(-(9*x-4)**2-(9*y-7)**2)

    return term1 + term2 + term3 + term4


z = franke_function(x,y)

#design matrix for x, second order polynomial
X1 = np.zeros((20,3))
X1[:,0] = 1

for i in range(20):
    X1[i,1] = x[i]
    X1[i,2] = x[i]**2

# manual inversion
beta = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(z_flat)

# scikit-learn
clf = skl.LinearRegression().fit(X1,y)
y_tilde = clf.predict(X1)

# MSE
MSE = mean_squared_error(y,y_tilde)

# R²
R_squared = r2_score(y,y_tilde)

print("MSE: %.2f" %MSE)
print("R_squared: %.2f" %R_squared)
#design matrix for y
