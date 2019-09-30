import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import regression as reg
import pandas as pd
import matplotlib.cm as cm

np.random.seed(14)
# make data points
x = np.arange(0,1,0.05)
y = np.arange(0,1,0.05)
x,y = np.meshgrid(x,y)
z = reg.franke_function(x,y)
z_flat = np.ravel(z)
z_flat = z_flat + np.random.normal(0, 1, z_flat.shape[0])



# file with all the parts concerning the franke function

#standard regression polynomials up to degree 5
maxdegree = 5

def OLS_analysis():
    print("Analysis for OLS")
    MSEs = np.zeros((maxdegree, 4))
    MSEs[:,0] = [i for i in range(1,maxdegree+1)]
    #print(MSEs)

    for degree in range(1,maxdegree+1):
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        beta = reg.OLS_fit(X,z_flat)
        z_tilde = reg.OLS_predict(beta, X)

        # MSE
        MSE = mean_squared_error(z_flat,z_tilde)
        MSEs[degree-1][1] = MSE
        # R²
        R_squared = r2_score(z_flat,z_tilde)

        print("Polynomial degree: %d" %degree)
        print("MSE standard: %.5f" %MSE)
        print("R_squared standard: %.5f" %R_squared)

        var_beta = np.linalg.inv(X.T.dot(X))

        # confidence interval for betas
        uppers = np.zeros(len(beta))
        lowers = np.zeros(len(beta))
        intervals = []
        for b in range(len(beta)):
            #sigma squared is 1
            var_b = var_beta[b][b]
            #print (var_b)
            upper = beta[b] + 1.96*np.sqrt(var_b)
            lower = beta[b] - 1.96*np.sqrt(var_b)
            uppers[b] = upper
            lowers[b] = lower
            intervals.append((lower,upper))
            #print("Confidence interval for beta_%d: [%g,%g]" %(b,lower, upper))

        # resampling of data
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)

        beta_train = reg.OLS_fit(X_train, z_train)
        z_tilde = reg.OLS_predict(beta_train, X_test)

        # MSE
        MSE = mean_squared_error(z_test,z_tilde)
        MSEs[degree-1][2] = MSE
        # R²
        R_squared = r2_score(z_test,z_tilde)

        print("MSE using data splitting: %.5f" %MSE)
        print("R_squared using data splitting: %.5f" %R_squared)

        # kfold cross validation
        MSE_kf = reg.k_fold_cross_validation(
            X,z_flat, 5, reg.OLS_fit, reg.OLS_predict)[0]
        print("MSE kfold: %.5f" %MSE_kf)
        MSEs[degree-1][3] = MSE_kf

        pcolormesh(MSE_kf)
        plt.show()

        if degree == maxdegree:
            plt.plot(beta, label='beta')
            plt.plot(uppers, label='upper boundary in CI')
            plt.plot(lowers, label = 'lower boundary in CI')
            #plt.errorbar(x=np.arange(0,len(beta)),
             #y=beta,
             #yerr=[(top-bot)/2 for top,bot in intervals],
             #fmt='o')
            #plt.hlines(xmin=0, xmax=25,
            #    y=np.mean(beta),
            #    linewidth=2.0,
            #    color="red")
            #plt.xaxis()
            plt.title('95 percent Confidence Interval for Beta when Degree=5')
            plt.ylabel('beta_i')
            plt.xlabel('i')
            plt.legend()
            plt.show()

    #intervals = []
    #fig, ax = plt.subplots()

    # hide axes
    #fig.patch.set_visible(False)
    #ax.axis('off')
    #ax.axis('tight')

    #df = pd.DataFrame(MSEs, columns=['Degree', 'No resampling', 'train_test_split', 'k-fold CV'])

#    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

#    fig.tight_layout()

#    plt.show()
        #plt.table(cellText = MSEs)
    #plt.show()
    print(MSEs)

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


def Lasso_analysis():
    lambdas = [10**i for i in range(-5,2)]
    degrees = [i for i in range(1,maxdegree+1)]
    MSEs_kfold = np.zeros((len(lambdas),maxdegree))


    #for degree in range(1, maxdegree+1):
    for degree in degrees:
        X = reg.CreateDesignMatrix_X(x,y,n=degree)
        print("degree: ", degree)
        #for lam in [10**i for i in range(-5,2)]:
        lam_index = 0
        for lam in lambdas:
            beta = reg.Lasso_fit(X,z_flat, lam)
            z_tilde = reg.Lasso_predict(beta, X)

            # MSE
            MSE = mean_squared_error(z_flat,z_tilde)
            # R²
            R_squared = r2_score(z_flat,z_tilde)

            MSE_kfold = reg.k_fold_cross_validation(
                X,z_flat, 5, reg.Lasso_fit, reg.Lasso_predict, _lambda=lam)[0]

            MSEs_kfold[lam_index][degree-1] = MSE_kfold
            lam_index+=1
            print("lambda value: ", lam)
            print("MSE standard: %.5f" %MSE)
            print("MSE kfold: %.5f" %MSE_kfold)
            print("R_squared standard: %.5f" %R_squared)

    #print(MSEs_kfold)
    fig, ax = plt.subplots()
    i = ax.imshow(MSEs_kfold, cmap=cm.jet, interpolation='nearest',extent=[1,5,-5,2])
    fig.colorbar(i)
    #ax = heatmap(MSEs_kfold)
    plt.show()



#Ridge_analysis()
#OLS_analysis()
Lasso_analysis()
#confidence interval for betas

#train test split

# k fold mse
