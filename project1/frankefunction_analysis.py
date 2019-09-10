import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score

#make data
x = np.arange(0,1,0.05)
y = np.arange(0,1,0.05)
x,y = np.meshgrid(x,y)

#franke function
def franke_function(x,y):
    term1 = 3/4*np.exp(-(9*x-2)**2/4 - (9*y-2)**2/4)
    term2 = 3/4*np.exp(-(9*x+1)**2/49 - (9*y+1)/10)
    term3 = 1/2*np.exp(-(9*x-7)**2/4-(9*y-3)**2/4)
    term4 = -1/5*np.exp(-(9*x-4)**2-(9*y-7)**2)

    return term1 + term2 + term3 + term4



z = franke_function(x,y) #+ np.random.randn(20,20)
z_flat = np.ravel(z)


x_flat = np.ravel(x)
y_flat = np.ravel(y)

# need to make a function for arbitrary order
# design matrix for order 2
#X = np.c_[np.ones(400),x_flat,y_flat,x_flat**2, y_flat**2, x_flat*y_flat]

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

X = CreateDesignMatrix_X(x,y,n=2)

# manual inversion
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_flat)


# scikit-learn
clf = skl.LinearRegression().fit(X,z_flat)
z_tilde = clf.predict(X)

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
    print("Confidence interval: [%g,%g]" %(lower, upper))


#copied from the internet, use for inspiration
def k_fold_cross_validation(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
