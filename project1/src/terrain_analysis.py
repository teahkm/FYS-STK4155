# parts f and g: real data

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import regression as reg
from sklearn.metrics import mean_squared_error, r2_score
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# fit polynomials up to degree 5
maxdegree = 5

# Load the terrain
terrain1 = imread('../Data/SRTM_data_Norway_1.tif')

# Downsample and normalize
terrain_downsized = imresize(terrain1, 0.05)
terrain_downsized = terrain_downsized/terrain_downsized.max()


# Make data
rows = np.linspace(0,1,terrain_downsized.shape[0])
cols = np.linspace(0,1,terrain_downsized.shape[1])

x, y = np.meshgrid(cols, rows)
z_flat = terrain_downsized.ravel()

# Show the terrain
def show_terrain():
    fig, (ax1, ax2) = plt.subplots(1,2)
    plt.suptitle('Terrain over Norway')
    ax1.imshow(terrain1, cmap='gray')
    ax2.imshow(terrain_downsized, cmap='gray')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Original data')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Downsampled data')
    plt.show()

# all the analysis of terrain data involving OLS
def OLS_analysis():
    print('Analysis for OLS')

    # matrices for table construction
    MSEs = np.zeros((maxdegree, 4))
    R2s = np.zeros((maxdegree, 3))

    # first column of tables: polynomial degrees
    degrees = [i for i in range(1,maxdegree+1)]
    MSEs[:,0] = degrees
    R2s[:,0] = degrees

    # for bias-variance tradeoff
    error_test = np.zeros(maxdegree)
    error_train = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)

    # find error as function of model complexity
    for degree in degrees:
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        # fitting without resampling
        beta = reg.OLS_fit(X,z_flat)
        z_tilde = reg.OLS_predict(beta, X)

        # MSE
        MSE = mean_squared_error(z_flat,z_tilde)
        MSEs[degree-1][1] = MSE
        # R²
        R_squared = r2_score(z_flat,z_tilde)
        R2s[degree-1][1] = R_squared

        #confidence interval of betas
        uppers, lowers = reg.CI_beta(X, beta)

        # fitting with resampling of data
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)

        beta_train = reg.OLS_fit(X_train, z_train)
        z_tilde = reg.OLS_predict(beta_train, X_test)

        # MSE
        MSE = mean_squared_error(z_test,z_tilde)
        MSEs[degree-1][2] = MSE
        # R²
        R_squared = r2_score(z_test,z_tilde)
        R2s[degree-1][2] = R_squared


        # fitting with k-fold cross validation
        MSE_kf = reg.k_fold_cross_validation(
                    X,z_flat, 5, reg.OLS_fit, reg.OLS_predict)[0]
        MSEs[degree-1][3] = MSE_kf

        # train vs test and bias/variance calculations using bootstrap
        e, e2, b, v = reg.bootstrap(X, z_flat, 100, fit_type=reg.OLS_fit, predict_type= reg.OLS_predict)
        error_test[degree-1] = e
        error_train[degree-1] = e2
        bias[degree-1] = b
        variance[degree-1] = v

        # train vs test using CV
        #e, e2 = reg.k_fold_cross_validation(X, z_flat, k=5, fit_type=reg.OLS_fit, predict_type= reg.OLS_predict, tradeoff = False)
        #error_test[degree-1] = e
        #error_train[degree-1] = e2

        # plot confidence interval for betas only for last degree
        if degree == maxdegree:
            sns.set();
            plt.fill_between(np.arange(len(beta)), uppers, lowers, label='95% confidence interval', color=(.5,.5,.5,.2))
            plt.plot(beta, label='beta', color="r")
            plt.title('95 percent Confidence Interval for Beta from OLS when Degree=5')
            plt.ylabel('beta_i')
            plt.xlabel('i')
            plt.legend()
            plt.show()


    # making LaTex table for MSEs
    df = pd.DataFrame(MSEs, columns=['Degree', 'No resampling', 'train_test_split', 'k-fold CV'])
    df['Degree'] = df['Degree'].astype(int)
    tab = df.to_latex(index=False, float_format="%.5f")
    print(f"\n\n{tab}\n\n")

    # making LaTex table for R2s
    df = pd.DataFrame(R2s, columns=['Degree', 'No resampling', 'train_test_split'])
    df['Degree'] = df['Degree'].astype(int)
    tab = df.to_latex(index=False, float_format="%.5f")
    print(f"\n\n{tab}\n\n")

    # plot test vs train MSE using bootstrap or CV
    sns.set();
    plt.plot(degrees, error_test, label='Test MSE')
    plt.plot(degrees, error_train, label='Train MSE')
    plt.title('Train vs Test MSE for OLS using the Bootstrap Method')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # plot bias-variance tradeoff using bootstrap
    sns.set();
    plt.plot(degrees, error_test, label='MSE')
    plt.plot(degrees, bias, label='bias')
    plt.plot(degrees, variance, label='Variance')
    plt.title('Bias-Variance Tradeoff for OLS using the Bootstrap Method')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


# all analysis of terrain data involving Ridge regression
def Ridge_analysis():
    print('Analysis for Ridge')

    lambdas = [10**i for i in range(-5,2)]  # lambda values to test
    degrees = [i for i in range(1,maxdegree+1)] # complexities to test

    MSEs = np.zeros((maxdegree,4)) # collection of MSEs for a given lambda
    MSEs[:,0] = degrees # first column in table is complexity

    R2s = np.zeros((maxdegree,3)) # collection of R"s for a given lambda
    R2s[:,0] = degrees # first column in table is complexity

    MSEs_kfold = np.zeros((len(lambdas),maxdegree)) # for plotting heatmap

    # for bias-variance tradeoff
    error_test = np.zeros(maxdegree)
    error_train = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)

    # find error as function of complexity and lambda
    for degree in range(1,maxdegree+1):

        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        lam_index = 0
        for lam in lambdas:

            MSE_kfold = reg.k_fold_cross_validation(
                        X,z_flat, 5, reg.Ridge_fit, reg.Ridge_predict, _lambda=lam)[0]

            MSEs_kfold[lam_index][degree-1] = MSE_kfold
            lam_index += 1


        # fitting without resampling chosen best lambda based on heatmap
        beta = reg.Ridge_fit(X,z_flat,_lambda=lambdas[0])
        z_tilde = reg.Ridge_predict(beta, X)

        # MSE
        MSE = mean_squared_error(z_flat,z_tilde)
        MSEs[degree-1][1] = MSE
        # R²
        R_squared = r2_score(z_flat,z_tilde)
        R2s[degree-1][1] = R_squared

        # fitting with resampling using best lambda based on k fold heatmap
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)
        beta_train = reg.Ridge_fit(X_train, z_train, _lambda=lambdas[0])
        z_tilde = reg.Ridge_predict(beta_train, X_test)

        # MSE
        MSE = mean_squared_error(z_test,z_tilde)
        MSEs[degree-1][2] = MSE

        # R²
        R_squared = r2_score(z_test,z_tilde)
        R2s[degree-1][2] = R_squared

        # kfold into table
        MSEs[degree-1][3] = MSEs_kfold[0][degree-1]

        # train vs test and bias/variance calculations using bootstrap for chosen lambda
        e, e2, b, v = reg.bootstrap(X, z_flat, 100, fit_type=reg.Ridge_fit, predict_type= reg.Ridge_predict, _lambda=lambdas[0])
        error_test[degree-1] = e
        error_train[degree-1] = e2
        bias[degree-1] = b
        variance[degree-1] = v

        # train vs test using CV
        #e, e2 = reg.k_fold_cross_validation(X, z_flat, k=5, fit_type=reg.Ridge_fit, predict_type= reg.Ridge_predict, tradeoff = False, _lambda=0.1)
        #error_test[degree-1] = e
        #error_train[degree-1] = e2


    # making LaTex table for MSEs (use the one corresponding to best lambda) to compare with OLS
    df = pd.DataFrame(MSEs, columns=['Degree', 'No resampling', 'train_test_split', 'k-fold CV'])
    df['Degree'] = df['Degree'].astype(int)
    tab = df.to_latex(index=False, float_format="%.5f")
    print(f"\n\n{tab}\n\n")

    # making LaTex table for R2s (use the one corresponding to best lambda) to compare with OLS
    df = pd.DataFrame(R2s, columns=['Degree', 'No resampling', 'train_test_split'])
    df['Degree'] = df['Degree'].astype(int)
    tab = df.to_latex(index=False, float_format="%.5f")
    print(f"\n\n{tab}\n\n")

    # plotting heatmap with kfold CV to choose best combination of degree and lambda
    fig, ax = plt.subplots()
    sns.heatmap(MSEs_kfold,xticklabels=degrees, yticklabels=np.log10(lambdas), annot=True, fmt='.2f')
    plt.xlabel('Polynomial degree')
    plt.ylabel('log(lambda)')
    plt.title('MSE as function of lambda and complexity with Ridge regression')
    plt.tight_layout()
    plt.show()

    # finding minimum MSE from k-fold
    print(MSEs_kfold.min())

    # plot test vs train MSE using bootstrap or CV for chosen lambda
    sns.set();
    plt.plot(degrees, error_test, label='Test MSE')
    plt.plot(degrees, error_train, label='Train MSE')
    plt.title('Train vs Test MSE for Ridge using 5-fold CV, with lambda=%.5f' %lambdas[0])
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # plot bias-variance tradeoff using bootstrap for chosen lambda
    sns.set();
    plt.plot(degrees, error_test, label='MSE')
    plt.plot(degrees, bias, label='bias')
    plt.plot(degrees, variance, label='Variance')
    plt.title('Bias-Variance Tradeoff for Ridge using Bootstrap, with lambda=%.5f' %lambdas[0])
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


# all analysis of terrain data involving Lasso regression
def Lasso_analysis():
    print('Analysis for Lasso')

    lambdas = [10**i for i in range(-5,2)]  # lambda values to test
    degrees = [i for i in range(1,maxdegree+1)] # complexities to test

    MSEs = np.zeros((maxdegree,4)) # collection of MSEs for a given lambda
    MSEs[:,0] = degrees # first column in table is complexity

    R2s = np.zeros((maxdegree,3)) # collection of R"s for a given lambda
    R2s[:,0] = degrees # first column in table is complexity

    MSEs_kfold = np.zeros((len(lambdas),maxdegree)) # for plotting heatmap


    # for bias-variance tradeoff
    error_test = np.zeros(maxdegree)
    error_train = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)

    # find error as function of complexity and lambda
    for degree in range(1,maxdegree+1):

        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        lam_index = 0
        for lam in lambdas:

            MSE_kfold = reg.k_fold_cross_validation(
                        X,z_flat, 5, reg.Lasso_fit, reg.Lasso_predict, _lambda=lam)[0]

            MSEs_kfold[lam_index][degree-1] = MSE_kfold
            lam_index += 1


        # fitting without resampling
        model = reg.Lasso_fit(X,z_flat,_lambda=1)
        z_tilde = reg.Lasso_predict(model, X)

        # MSE
        MSE = mean_squared_error(z_flat,z_tilde)
        MSEs[degree-1][1] = MSE
        # R²
        R_squared = r2_score(z_flat,z_tilde)
        R2s[degree-1][1] = R_squared

        # fitting with resampling using a chosen lambda
        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)
        model_train = reg.Lasso_fit(X_train, z_train, _lambda=1)
        z_tilde = reg.Lasso_predict(model_train, X_test)

        # MSE
        MSE = mean_squared_error(z_test,z_tilde)
        MSEs[degree-1][2] = MSE

        # R²
        R_squared = r2_score(z_test,z_tilde)
        R2s[degree-1][2] = R_squared

        # kfold into table
        MSEs[degree-1][3] = MSEs_kfold[6][degree-1]

        # train vs test and bias/variance calculations using bootstrap for chosen lambda
        e, e2, b, v = reg.bootstrap(X, z_flat, 100, fit_type=reg.Lasso_fit, predict_type= reg.Lasso_predict, _lambda=1)
        error_test[degree-1] = e
        error_train[degree-1] = e2
        bias[degree-1] = b
        variance[degree-1] = v

        # train vs test using CV
        #e, e2 = reg.k_fold_cross_validation(X, z_flat, k=5, fit_type=reg.Lasso_fit, predict_type= reg.Lasso_predict, tradeoff = False, _lambda=1)
        #error_test[degree-1] = e
        #error_train[degree-1] = e2

    # find mimimum MSE from k-fold
    print(MSEs_kfold.min())

    # making LaTex table for MSEs (use the one corresponding to best lambda) to compare with OLS
    df = pd.DataFrame(MSEs, columns=['Degree', 'No resampling', 'train_test_split', 'k-fold CV'])
    df['Degree'] = df['Degree'].astype(int)
    tab = df.to_latex(index=False, float_format="%.5f")
    print(f"\n\n{tab}\n\n")

    # making LaTex table for R2s (use the one corresponding to best lambda) to compare with OLS
    df = pd.DataFrame(R2s, columns=['Degree', 'No resampling', 'train_test_split'])
    df['Degree'] = df['Degree'].astype(int)
    tab = df.to_latex(index=False, float_format="%.5f")
    print(f"\n\n{tab}\n\n")

    # plotting heatmap with kfold CV to choose best combination of degree and lambda
    fig, ax = plt.subplots()
    sns.heatmap(MSEs_kfold,xticklabels=degrees, yticklabels=np.log10(lambdas), annot=True, fmt='.2f')
    plt.xlabel('Polynomial degree')
    plt.ylabel('log(lambda)')
    plt.title('MSE as function of lambda and complexity with Lasso regression')
    plt.tight_layout()
    plt.show()

    # plot test vs train MSE using bootstrap or CV for chosen lambda
    sns.set();
    plt.plot(degrees, error_test, label='Test MSE')
    plt.plot(degrees, error_train, label='Train MSE')
    plt.title('Train vs Test MSE for Lasso using the Bootstrap Method')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # plot bias-variance tradeoff using bootstrap for chosen lambda
    sns.set();
    plt.plot(degrees, error_test, label='MSE')
    plt.plot(degrees, bias, label='bias')
    plt.plot(degrees, variance, label='Variance')
    plt.title('Bias-Variance Tradeoff for Lasso using the Bootstrap Method')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


# plotting the surface of data
def plot_surface(surface, title):
    sns.set()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(y, x, surface, cmap=cm.Greens, linewidth=0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()


# plotting the chosen model
def plot_best_fit():
    X = reg.CreateDesignMatrix_X(x,y,n=5)
    beta = reg.OLS_fit(X,z_flat)
    z_tilde = reg.OLS_predict(beta, X)
    z_tilde = z_tilde.reshape(terrain_downsized.shape)
    plot_surface(z_tilde, 'Surface plot of OLS fifth order polynomial')


#show_terrain()
#OLS_analysis()
#Ridge_analysis()
#Lasso_analysis()
#plot_surface(terrain_downsized, 'Surface plot of downsampled terrain')
#plot_best_fit()
