import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, cross_validate, KFold


from common import franke_function, x, y, CreateDesignMatrix_X

#try with noise
z = franke_function(x,y) #+ np.random.randn(20,20)
z_flat = np.ravel(z)


x_flat = np.ravel(x)
y_flat = np.ravel(y)

X = CreateDesignMatrix_X(x,y,n=2)


# The Ridge regression with a hyperparameter lambda = 0.1
_lambda = 0.1
#clf_ridge = skl.Ridge(alpha=_lambda).fit(X, z_flat)
#z_ridge = clf_ridge.predict(X)

def Ridge_fit(X, y, _lambda=0.1):
    XtX = X.T.dot(X) #help variable
    beta = np.linalg.inv(XtX + _lambda*np.identity(len(XtX[0]))).dot(X.T).dot(y)
    return beta

def Ridge_predict(beta, X):
    return X.dot(beta)

beta = Ridge_fit(X, z_flat, 0.1)
z_ridge = Ridge_predict(beta, X)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(z_flat, z_ridge))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(z_flat, z_ridge))
# Mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(z_flat, z_ridge))
#print(clf_ridge.coef_, clf_ridge.intercept_)


X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)
beta_train = Ridge_fit(X_train, z_train)
z_tilde = Ridge_predict(beta_train, X_test)

# MSE
MSE = mean_squared_error(z_test,z_tilde)

# R²
R_squared = r2_score(z_test,z_tilde)
print("MSE: %.2f" %MSE)
print("R_squared: %.2f" %R_squared)


def k_fold_cross_validation(X, z, k=5):
    MSE = []

    #shuffling data
    index = np.random.permutation(np.arange(len(z)))
    X_shuffled = X[index]
    z_shuffled = z[index]

    #split into folds
    i = np.arange(len(z)) % k
    for fold in range(k):
        X_test = X_shuffled[i==fold]
        X_train = X_shuffled[i!=fold]
        z_test = z_shuffled[i==fold]
        z_train = z_shuffled[i!=fold]

        #fit model
        beta = Ridge_fit(X_train, z_train)
        z_tilde = Ridge_predict(beta_train, X_test)

        #evaluate MSE
        MSE.append(mean_squared_error(z_test,z_tilde))

    CV_score = np.mean(MSE)
    return CV_score

print(k_fold_cross_validation(X, z_flat, 5))
cv = KFold(n_splits=5, shuffle=True)
comparison = cross_validate(
    skl.Ridge(),
    X,
    z_flat,
    cv=cv,
    scoring=make_scorer(mean_squared_error)
)

print(np.mean(comparison["test_score"]))
