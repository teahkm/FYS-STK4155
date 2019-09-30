import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def franke_function(x,y):
    """
	Test function.

    Args:
        x (matrix or flattened matrix): input values
        y (matrix or flattened matrix): input values

    Returns:
        f(x,y)
	"""
    term1 = 3/4*np.exp(-(9*x-2)**2/4 - (9*y-2)**2/4)
    term2 = 3/4*np.exp(-(9*x+1)**2/49 - (9*y+1)/10)
    term3 = 1/2*np.exp(-(9*x-7)**2/4-(9*y-3)**2/4)
    term4 = -1/5*np.exp(-(9*x-4)**2-(9*y-7)**2)

    return term1 + term2 + term3 + term4

def CreateDesignMatrix_X(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X

def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))


def OLS_fit(X,y):
    """ A function that caluculates the coefficients beta_i in ordinary least squares.

        Args:
            X (matrix): design matrix
            y (array): true values to be fitted

        Returns:
            beta (array): regression coefficients
    """
    #pseudo inverse, SVD
    #A = np.linalg.pinv(X)
    inv = SVDinv(X.T.dot(X))
    #beta = A.dot(y)
    beta = inv.dot(X.T).dot(y)
    return beta

def OLS_predict(beta, X):
    """ A function that predicts a value using ordinary least squares.

        Args:
            beta (array): regression coefficients
            X (matrix): design matrix

        Returns:
            predicted value of X times beta
    """
    return X.dot(beta)

def Ridge_fit(X, y, _lambda=0.1):
    """ A function that calculates the coefficients beta_i using Ridge regression.

        Args:
            X (matrix): design matrix
            y (array): true values to be fitted
            _lambda (float): tuning parameter

        Returns:
            beta (array): regression coefficients
    """

    XtX = X.T.dot(X) #help variable
    inv = SVDinv(XtX + _lambda*np.identity(len(XtX[0])))
    #beta = np.linalg.inv(XtX + _lambda*np.identity(len(XtX[0]))).dot(X.T).dot(y)
    beta = inv.dot(X.T).dot(y)
    return beta

def Ridge_predict(beta, X):
    """ A function that predicts a value using linear regression.

        Args:
            beta (array): regression coefficients
            X (matrix): design matrix

        Returns:
            predicted value of X times beta
    """
    return X.dot(beta)

def Lasso_fit(X, y, _lambda=0.1):
    clf = Lasso(alpha=_lambda)
    clf.fit(X, y)
    #beta = np.zeros(len(clf.coef_)+1)
    beta = clf.coef_
    #beta[0] = clf.intercept_
    #beta[1:] = clf.coef_
    return beta.T #clf

def Lasso_predict(beta, X):
    return X.dot(beta)




def k_fold_cross_validation(X, z, k=5, fit_type=OLS_fit, predict_type= OLS_predict, tradeoff = False, **parameters):
    """ A function that predicts a value using linear regression.

        Args:
            X (matrix): design matrix
            z (array): true values
            k (int): number of folds
            fit_type (function): which regression fit to use
            predict_type (function): which regression to use for prediction


        Returns:
            tot_err (float): average mean squared error after k folds
            tot_bias (float): average bias after k folds
            tot_var (float): average variance of predicted values after k folds
    """
    MSE_test = []
    MSE_train = []
    bias = []
    variance = []

    # shuffling data
    index = np.random.permutation(np.arange(len(z)))
    X_shuffled = X[index]
    z_shuffled = z[index]

    # split into folds


    if not tradeoff:
        i = np.arange(len(z)) % k

        for fold in range(k):
            X_test = X_shuffled[i==fold]
            X_train = X_shuffled[i!=fold]
            z_test = z_shuffled[i==fold]
            z_train = z_shuffled[i!=fold]

            # fit model
            beta_train = fit_type(X_train, z_train, **parameters)
            z_tilde_test = predict_type(beta_train, X_test)
            z_tilde_train = predict_type(beta_train, X_train)

            # evaluate MSE, bias, variance
            MSE_test.append(np.mean((z_test - z_tilde_test)**2))
            MSE_train.append(np.mean((z_train - z_tilde_train)**2))
            #bias.append(np.mean((z_test - np.mean(z_tilde_test))**2))
            #variance.append(np.var(z_tilde_test))
            return MSE_test, MSE_train

    else:
        X_shuf_train, X_shuf_test, z_shuf_train, z_shuf_test = train_test_split(
                                                                X_shuffled, z_shuffled, test_size = 0.33)

        z_all_preds = np.empty((z_shuf_test.shape[0], k))
        #print("all preds: ", z_all_preds.shape)

        i = np.arange(len(z_shuf_train)) % k

        for fold in range(k):
            X_test = X_shuf_train[i==fold]
            X_train = X_shuf_train[i!=fold]
            z_test = z_shuf_train[i==fold]
            z_train = z_shuf_train[i!=fold]

            # fit model
            beta_train = fit_type(X_train, z_train, **parameters)
            z_tilde_test = predict_type(beta_train, X_shuf_test)
            #print("z tilde test: ", z_tilde_test.shape)
            #z_tilde_train = predict_type(beta_train, X_train)
            z_all_preds[:, fold] = z_tilde_test

            # evaluate MSE, bias, variance
            #MSE_test.append(np.mean((z_test - z_tilde_test)**2))
            #MSE_train.append(np.mean((z_train - z_tilde_train)**2))
            #bias.append(np.mean((z_test - np.mean(z_tilde_test))**2))
            #variance.append(np.var(z_tilde_test))

    # average the results
        print("z_test: ", z_shuf_test.shape)
        print("z_pred: ", z_all_preds.shape)
        tot_err_test = np.mean((z_shuf_test - np.mean(z_all_preds, axis=1))**2 )
        tot_bias = np.mean( (z_shuf_test - np.mean(z_all_preds, axis=1, keepdims=True))**2 )
        tot_var = np.mean( np.var(z_all_preds, axis=1) )

        return tot_err_test, tot_bias, tot_var

    """MSE_train = []
    MSE_test = []
    bias = []
    variance = []

    # shuffling data
    index = np.random.permutation(np.arange(len(z)))
    X_shuffled = X[index]
    z_shuffled = z[index]

    # split into folds
    i = np.arange(len(z)) % k
    print(i.shape)
    z_pred = np.empty((z[i==1].shape[0], k))
    z_test_arr = np.empty((z[i==1].shape[0], k))
    for fold in range(k):
        X_test = X_shuffled[i==fold]
        X_train = X_shuffled[i!=fold]
        z_test = z_shuffled[i==fold]
        z_train = z_shuffled[i!=fold]

        z_test_arr[:, fold] = z_test

        # fit model
        beta_train = fit_type(X_train, z_train, **parameters)
        z_tilde =predict_type(beta_train, X_test).ravel()
        print(predict_type(beta_train, X_test).shape)
        print(z_pred.shape)
        z_pred[:, fold] = z_tilde

        # evaluate MSE, bias, variance
        MSE_test.append(np.mean((z_test - z_tilde)**2))
        MSE_train.append(np.mean((z_train - z_tilde)**2))
        #bias.append(np.mean((z_test - np.mean(z_tilde))**2))
        #variance.append(np.var(z_tilde))

    # average the results
    #MSE_test = np.mean( np.mean((z_test_arr - z_pred)**2, axis=1) )
    bias = np.mean( (z_test_arr - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1) )
    #tot_err = np.mean(MSE)
    #tot_bias = np.mean(bias)
    #tot_var = np.mean(variance)

    return MSE_test, MSE_train, bias, variance
    #return tot_err, tot_bias, tot_var
"""