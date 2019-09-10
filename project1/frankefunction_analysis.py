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
z_flat = np.ravel(z)

#def design_matrix_degree_p(x,y,p):
#    X = np.c_(np.ones(len(x)))
#    return X
x_flat = np.ravel(x)
y_flat = np.ravel(y)

X = np.c_[np.ones(400),x_flat,y_flat,x_flat**2, y_flat**2, x_flat*y_flat]


# manual inversion
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_flat)


# scikit-learn
clf = skl.LinearRegression().fit(X,z_flat)
z_tilde = clf.predict(X)

# MSE
MSE = mean_squared_error(z_flat,z_tilde)

# R²
R_squared = r2_score(z_flat,z_tilde)

print("MSE: %.2f" %MSE)
print("R_squared: %.2f" %R_squared)
