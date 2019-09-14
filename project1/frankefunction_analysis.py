import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score


from common import franke_function, x, y, CreateDesignMatrix_X


z = franke_function(x,y) #+ np.random.randn(20,20)
z_flat = np.ravel(z)


x_flat = np.ravel(x)
y_flat = np.ravel(y)


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
