"""
 
     random_forest example 
     first running the data_manipulation.py to prepare the data. 


"""
from sklearn import ensemble

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	X_train, y_train, test_size=0.2, random_state=0)


random_forest = ensemble.RandomForestClassifier(n_estimators = 200, 
	n_jobs = 20,
	random_state=4)
random_forest.fit(X = X_train, y=y_train)
quadratic_weighted_kappa(random_forest.predict(X = X_test), y_test)

train_RF = []
test_RF = []
N = np.arange(20,300,10)
for n in N:
	print 'begain training process with ' + str(n) + ' tress'
	random_forest = ensemble.RandomForestClassifier(n_estimators = n,
		n_jobs = 20,
		random_state=4)
	random_forest.fit(X = X_train, y=y_train)
	train_RF.append(quadratic_weighted_kappa(random_forest.predict(X = X_train), y_train))
	test_RF.append(quadratic_weighted_kappa(random_forest.predict(X = X_test), y_test))
	print 'this process finished'


