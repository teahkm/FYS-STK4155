# parts f and g: real data

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import regression as reg
from sklearn.metrics import mean_squared_error, r2_score
from scipy.misc import imresize

maxdegree = 5

# Load the terrain
terrain1 = imread('Data/SRTM_data_Norway_1.tif')

# Downsample
terrain_downsized = imresize(terrain1, 0.05)

#linspace between 0 1 with number of points
rows = np.linspace(0,1,terrain_downsized.shape[0])
cols = np.linspace(0,1,terrain_downsized.shape[1])
#rows = np.arange(terrain_downsized.shape[0])
#cols = np.arange(terrain_downsized.shape[1])

x, y = np.meshgrid(cols, rows)
z_flat = terrain_downsized.ravel()

# Show the terrain
def show_terrain():
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain_downsized, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def OLS_analysis():
    print("Analysis for OLS")

    #MSEs = np.zeros((maxdegree, 4))
    #MSEs[:,0] = [i for i in range(1,maxdegree+1)]
    #print(MSEs)

    for degree in range(1,maxdegree+1):
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        beta = reg.OLS_fit(X,z_flat)
        z_tilde = reg.OLS_predict(beta, X)

        # MSE
        MSE = mean_squared_error(z_flat,z_tilde)
        #MSEs[degree-1][1] = MSE
        # R²
        R_squared = r2_score(z_flat,z_tilde)

        MSE_kfold = reg.k_fold_cross_validation(X, z_flat, k=10, fit_type=reg.OLS_fit, predict_type= reg.OLS_predict, tradeoff = False)[0]
        #MSE_boot = reg.bootstrap(X, z_flat, 100, fit_type=reg.OLS_fit, predict_type= reg.OLS_predict)[0]

        print("Polynomial degree: %d" %degree)
        print("MSE standard: %.5f" %MSE)
        print("MSE k_fold: %.5f" %MSE_kfold)
        #print("MSE boot: %.5f" %MSE_boot)
        print("R_squared standard: %.5f" %R_squared)

def Ridge_analysis():
    print("Analysis for Ridge")

    for degree in range(1, maxdegree+1):
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        for lam in [10**i for i in range(-5,2)]:

            MSE_kfold = reg.k_fold_cross_validation(X, z_flat, k=10, fit_type=reg.Ridge_fit, predict_type= reg.Ridge_predict, tradeoff = False, _lambda=lam)[0]

            print("Polynomial degree: %d" %degree)
            print("Lambda value: %.5f" %lam)
            print("MSE k_fold: %.5f" %MSE_kfold)

def Lasso_analysis():
    print("Analysis for Ridge")

    for degree in range(1, maxdegree+1):
        X = reg.CreateDesignMatrix_X(x,y,n=degree)

        for lam in [10**i for i in range(-5,2)]:

            MSE_kfold = reg.k_fold_cross_validation(X, z_flat, k=10, fit_type=reg.Lasso_fit, predict_type= reg.Lasso_predict, tradeoff = False, _lambda=lam)[0]

            print("Polynomial degree: %d" %degree)
            print("Lambda value: %.5f" %lam)
            print("MSE k_fold: %.5f" %MSE_kfold)

#show_terrain()
#OLS_analysis()
Ridge_analysis()
#Lasso_analysis()
