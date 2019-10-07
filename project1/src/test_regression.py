import numpy as np
import regression as reg
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate

# make data
x = np.arange(0,1,0.05)
y = np.arange(0,1,0.05)
x,y = np.meshgrid(x,y)
z = reg.franke_function(x,y)
z_flat = np.ravel(z)

X = reg.CreateDesignMatrix_X(x,y,n=5)

def test_MSE_wo_resamp_OLS():
    ''' Function testing the OLS code without resampling by comparing to scikit-learn. '''

    beta = reg.OLS_fit(X,z_flat)
    z_tilde = reg.OLS_predict(beta, X)
    MSE_own = mean_squared_error(z_flat,z_tilde)

    model = skl.LinearRegression()
    model.fit(X,z_flat)
    z_tilde_skl = model.predict(X)
    MSE_skl = mean_squared_error(z_flat,z_tilde_skl)

    tol = 1E-14
    success = abs(MSE_own - MSE_skl) < tol
    msg = 'Error: The MSE for own OLS code did not match the MSE from scikit-learn.'
    assert success, msg

def test_MSE_wo_resamp_Ridge():
    ''' Function testing the Ridge code without resampling by comparing to scikit-learn. '''

    beta = reg.Ridge_fit(X,z_flat,0.1)
    z_tilde = reg.Ridge_predict(beta, X)
    MSE_own = mean_squared_error(z_flat,z_tilde)

    model = skl.Ridge(alpha=0.1, fit_intercept=False)
    model.fit(X,z_flat)
    z_tilde_skl = model.predict(X)
    MSE_skl = mean_squared_error(z_flat,z_tilde_skl)

    tol = 1E-14
    success = abs(MSE_own - MSE_skl) < tol
    msg = 'Error: The MSE for own Ridge code did not match the MSE from scikit-learn.'
    assert success, msg

def test_MSE_wo_resamp_Lasso():
    ''' Function testing the Lasso code without resampling by comparing to scikit-learn. '''

    beta = reg.Lasso_fit(X,z_flat,0.1)
    z_tilde = reg.Lasso_predict(beta, X)
    MSE_own = mean_squared_error(z_flat,z_tilde)

    model = skl.Lasso(alpha=0.1)
    model.fit(X[:,1:],z_flat)
    z_tilde_skl = model.predict(X[:,1:])
    MSE_skl = mean_squared_error(z_flat,z_tilde_skl)

    tol = 1E-14
    success = abs(MSE_own - MSE_skl) < tol
    msg = 'Error: The MSE for own Lasso code did not match the MSE from scikit-learn.'
    assert success, msg

def test_MSE_w_kfold_OLS():
    ''' Function testing the k-fold code on OLS by comparing to scikit-learn. '''

    MSE_kf = reg.k_fold_cross_validation(
        X,z_flat, 5, reg.OLS_fit, reg.OLS_predict)[0]

    cv = KFold(n_splits=5, shuffle=True)
    comparison = cross_validate(
        skl.LinearRegression(),
        X,
        z_flat,
        cv=cv,
        scoring=make_scorer(mean_squared_error)
    )

    MSE_kf_skl = np.mean(comparison["test_score"])

    tol = 1E-3
    success = abs(MSE_kf - MSE_kf_skl) < tol
    msg = 'Error: The MSE from own k-fold code for OLS did not match the MSE from scikit-learn.'
    assert success, msg

def test_MSE_w_kfold_Ridge():
    ''' Function testing the k-fold code on Ridge by comparing to scikit-learn. '''

    MSE_kf = reg.k_fold_cross_validation(
        X,z_flat, 5, reg.Ridge_fit, reg.Ridge_predict, 0.01)[0]

    cv = KFold(n_splits=5, shuffle=True)
    comparison = cross_validate(
        skl.Ridge(alpha=0.01, fit_intercept=False),
        X,
        z_flat,
        cv=cv,
        scoring=make_scorer(mean_squared_error)
    )

    MSE_kf_skl = np.mean(comparison["test_score"])

    tol = 1E-2
    success = abs(MSE_kf - MSE_kf_skl) < tol
    msg = 'Error: The MSE from own k-fold code for Ridge did not match the MSE from scikit-learn.'
    assert success, msg

def test_MSE_w_kfold_Lasso():
    ''' Function testing the k-fold code on Lasso by comparing to scikit-learn. '''

    MSE_kf = reg.k_fold_cross_validation(
        X,z_flat, 5, reg.Lasso_fit, reg.Lasso_predict, 0.01)[0]

    cv = KFold(n_splits=5, shuffle=True)
    comparison = cross_validate(
        skl.Lasso(alpha=0.01),
        X[:,1:],
        z_flat,
        cv=cv,
        scoring=make_scorer(mean_squared_error)
    )

    MSE_kf_skl = np.mean(comparison["test_score"])

    tol = 1E-2
    success = abs(MSE_kf - MSE_kf_skl) < tol
    msg = 'Error: The MSE from own k-fold code for Lasso did not match the MSE from scikit-learn.'
    assert success, msg
