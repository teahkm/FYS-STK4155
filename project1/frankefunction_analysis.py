import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score


from common import franke_function, x, y, CreateDesignMatrix_X
from regression import OLS_fit, OLS_predict

#try with noise
z = franke_function(x,y) #+ np.random.randn(20,20)
z_flat = np.ravel(z)

X = CreateDesignMatrix_X(x,y,n=2)

"""
def OLS_fit(X,y):
    #pseudo inverse, SVD
    A = np.linalg.pinv(X)
    beta = A.dot(y)
    #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return beta

def OLS_predict(beta, X):
    return X.dot(beta)
"""

def main():
    # scikit-learn
    #clf = skl.LinearRegression().fit(X,z_flat)
    #z_tilde = clf.predict(X)

    #beta, z_tilde = OLS(X, z_flat)
    beta = OLS_fit(X,z_flat)
    z_tilde = OLS_predict(beta, X)

    # MSE
    MSE = mean_squared_error(z_flat,z_tilde)

    # R²
    R_squared = r2_score(z_flat,z_tilde)

    print("MSE: %.2f" %MSE)
    print("R_squared: %.2f" %R_squared)


    # variance of betas
    var_beta = np.linalg.inv(X.T.dot(X))

    # confidence interval for betas
    for b in range(len(beta)):
        #sigma squared is 1
        var_b = var_beta[b][b]
        #print (var_b)
        upper = beta[b] + 1.96*np.sqrt(var_b)
        lower = beta[b] - 1.96*np.sqrt(var_b)
        print("Confidence interval for beta_%d: [%g,%g]" %(b,lower, upper))


if __name__ == "__main__":
    main()
