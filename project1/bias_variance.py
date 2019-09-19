import matplotlib.pyplot as plt
import numpy as np
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.preprocessing import PolynomialFeatures
from common import x,y, franke_function, CreateDesignMatrix_X
from frankefunction_analysis import OLS_fit, OLS_predict
from cross_validation import k_fold_cross_validation
np.random.seed(2018)

maxdegree = 5


#arrays for plotting
error = np.zeros(maxdegree+1)
bias = np.zeros(maxdegree+1)
variance = np.zeros(maxdegree+1)
polydegree = np.zeros(maxdegree+1)

z = franke_function(x,y)

for degree in range(maxdegree+1):
    X = CreateDesignMatrix_X(x,y,n=degree)

    # find error, bias and variance from CV
    e, b, v = k_fold_cross_validation(X, z, k=5)

    polydegree[degree] = degree
    error[degree] = e
    bias[degree] = b
    variance[degree] = v
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
