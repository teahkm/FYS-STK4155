import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
        print("MSE standard: %.5f" %MSE)
        print("R_squared standard: %.5f" %R_squared)

        var_beta = np.linalg.inv(X.T.dot(X))

        # confidence interval for betas
        for b in range(len(beta)):
            #sigma squared is 1
            var_b = var_beta[b][b]
            #print (var_b)
            upper = beta[b] + 1.96*np.sqrt(var_b)
            lower = beta[b] - 1.96*np.sqrt(var_b)
            print("Confidence interval for beta_%d: [%g,%g]" %(b,lower, upper))

        # resampling of data
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)

        beta_train = reg.OLS_fit(X_train, z_train)
        z_tilde = reg.OLS_predict(beta_train, X_test)

        # MSE
        MSE = mean_squared_error(z_test,z_tilde)
        # R²
        R_squared = r2_score(z_test,z_tilde)

        print("MSE using data splitting: %.5f" %MSE)
        print("R_squared using data splitting: %.5f" %R_squared)

        # kfold cross validation
        print("MSE kfold: %.5f" %reg.k_fold_cross_validation(
            X,z_flat, 5, reg.OLS_fit, reg.OLS_predict)[0])

def Ridge_analysis():
    lambdas = [10**i for i in range(-5,5)]
    print(lambdas)
    error = np.zeros(len(lambdas))
    R_2 = np.zeros(len(lambdas))

    print("Analysis for Ridge")

    for degree in range(maxdegree+1):
        print("Polynomial degree: %d" %degree)

        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        for i in range(len(lambdas)):
            #beta = reg.Ridge_fit(X,z_flat, lambdas[i])
            #z_tilde = reg.Ridge_predict(beta, X)

            #clf = skl.Ridge(alpha=lambdas[i]).fit(X,z_flat)
            #z_tilde_comp = clf.predict(X)

            # MSE
            #MSE = mean_squared_error(z_flat,z_tilde)
            #MSE_comp = mean_squared_error(z_flat, z_tilde_comp)
            #error[i] = mean_squared_error(z_flat,z_tilde)
            # R²
            #R_squared = r2_score(z_flat,z_tilde)
            #R_squared_comp = r2_score(z_flat, z_tilde_comp)
            #R_2[i] = r2_score(z_flat,z_tilde)

            error[i] = reg.k_fold_cross_validation(
                X,z_flat, 5, reg.Ridge_fit, reg.Ridge_predict, _lambda=lambdas[i])[0]

        plt.plot(np.log10(lambdas), error, label="MSE")
        #plt.plot(np.log10(lambdas), R_2, label="R2")
        plt.xlabel("lambda")
        plt.ylabel("error")
        plt.legend()
        plt.show()

        # data splitting
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)
        beta_train = reg.Ridge_fit(X_train, z_train, _lambda=0.001)
        z_tilde = reg.Ridge_predict(beta_train, X_test)

        # MSE
        MSE = mean_squared_error(z_test,z_tilde)

        # R²
        R_squared = r2_score(z_test,z_tilde)
        print("MSE data split: %.2f" %MSE)
        print("R_squared data split: %.2f" %R_squared)

            #print("Lambda value: %.5f" %lambdas[i])
            #print("MSE: %.6f" %MSE)
            #print("MSE sklearn: %.6f" %MSE_comp)
            #print("R_squared: %.6f" %R_squared)
            #print("R_2 sklearn: %.6f" %R_squared_comp)

        # k fold cross validation
        print("MSE kfold: %.5f" %reg.k_fold_cross_validation(
            X,z_flat, 5, reg.Ridge_fit, reg.Ridge_predict, _lambda=0.001)[0])


Ridge_analysis()
#OLS_analysis()
#confidence interval for betas

#train test split

# k fold mse
