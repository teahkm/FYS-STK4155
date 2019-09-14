import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
from common import franke_function, x, y, CreateDesignMatrix_X
from sklearn.model_selection import train_test_split

#train_test_split
#r2 og MSE på test
#cross-validation
#compare with scitkit-learn

z = franke_function(x,y)
z_flat = np.ravel(z)
X = CreateDesignMatrix_X(x,y,n=2)

X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)

# scikit-learn
clf = skl.LinearRegression().fit(X_train, z_train)
z_tilde = clf.predict(X_test)

# MSE
MSE = mean_squared_error(z_test,z_tilde)

# R²
R_squared = r2_score(z_test,z_tilde)

print("MSE: %.2f" %MSE)
print("R_squared: %.2f" %R_squared)
