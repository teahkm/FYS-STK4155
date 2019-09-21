import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
import regression as reg

# make data points
x = np.arange(0,1,0.05)
y = np.arange(0,1,0.05)
x,y = np.meshgrid(x,y) #+ np.random.randn(20,20)
z = reg.franke_function(x,y)
z_flat = np.ravel(z)


# file with all the parts concerning the franke function

#standard regression polynomials up to degree 5
maxdegree = 5

def OLS_analysis():
    print("Analysis for OLS")

    for degree in range(maxdegree+1):
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        beta = reg.OLS_fit(X,z_flat)
        z_tilde = reg.OLS_predict(beta, X)

        # MSE
        MSE = mean_squared_error(z_flat,z_tilde)
        # R²
        R_squared = r2_score(z_flat,z_tilde)

        print("Polynomial degree: %d" %degree)
        print("MSE: %.2f" %MSE)
        print("R_squared: %.2f" %R_squared)

        var_beta = np.linalg.inv(X.T.dot(X))

        # confidence interval for betas
        for b in range(len(beta)):
            #sigma squared is 1
            var_b = var_beta[b][b]
            #print (var_b)
            upper = beta[b] + 1.96*np.sqrt(var_b)
            lower = beta[b] - 1.96*np.sqrt(var_b)
            print("Confidence interval for beta_%d: [%g,%g]" %(b,lower, upper))

def Ridge_analysis():
    lambdas = [10**-i for i in range(5)]
    print("Analysis for Ridge")

    for degree in range(maxdegree+1):
        print("Polynomial degree: %d" %degree)
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        for lam in lambdas:
            beta = reg.Ridge_fit(X,z_flat, lam)
            z_tilde = reg.Ridge_predict(beta, X)

            # MSE
            MSE = mean_squared_error(z_flat,z_tilde)
            # R²
            R_squared = r2_score(z_flat,z_tilde)

            print("Lambda value: %.5f" %lam)
            print("MSE: %.2f" %MSE)
            print("R_squared: %.2f" %R_squared)

Ridge_analysis()
#confidence interval for betas

#train test split

# k fold mse
