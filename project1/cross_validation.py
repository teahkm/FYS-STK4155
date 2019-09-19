import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from common import franke_function, x, y, CreateDesignMatrix_X
from sklearn.model_selection import train_test_split, cross_validate, KFold
from frankefunction_analysis import OLS_fit, OLS_predict

#train_test_split
#r2 og MSE på test
#cross-validation
#compare with scitkit-learn

z = franke_function(x,y)
z_flat = np.ravel(z)
X = CreateDesignMatrix_X(x,y,n=2)

X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.33)

# use own code here, FIX THIS
#clf = skl.LinearRegression().fit(X_train, z_train)
#z_tilde = clf.predict(X_test)
beta_train = OLS_fit(X_train, z_train)
z_tilde = OLS_predict(beta_train, X_test)

# MSE
MSE = mean_squared_error(z_test,z_tilde)

# R²
R_squared = r2_score(z_test,z_tilde)

print("MSE: %.2f" %MSE)
print("R_squared: %.2f" %R_squared)

def k_fold_cross_validation(X, z, k=5):
    MSE = []
    bias = []
    variance = []

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
        beta_train = OLS_fit(X_train, z_train)
        z_tilde = OLS_predict(beta_train, X_test)

        #evaluate MSE
        MSE.append(np.mean((z_test - z_tilde)**2))
        bias.append((z_test - np.mean(z_tilde))**2)
        variance.append(np.var(z_tilde))

    tot_err = np.mean(MSE)
    tot_bias = np.mean(bias)
    tot_var = np.mean(variance)

    return tot_err, tot_bias, tot_var

# comparing results to scikit-learn
print(k_fold_cross_validation(X,z_flat, 5))
cv = KFold(n_splits=5, shuffle=True)
comparison = cross_validate(
    skl.LinearRegression(),
    X,
    z_flat,
    cv=cv,
    scoring=make_scorer(mean_squared_error)
)

print(np.mean(comparison["test_score"]))
