""" 

         first run the data_manipulation.py to prepare data 
         require sk-learn


"""

adaboost = ensemble.AdaBoostClassifier(n_estimators = 200, 
	learning_rate = 1)
adaboost.fit(X = X_train, y = y_train)
quadratic_weighted_kappa(adaboost.predict(X_test), y_test)

N = np.arange(20,400,20)
ada_train = []
ada_test = []
for n in N: 
	print 'begin'
	adaboost = ensemble.AdaBoostClassifier(n_estimators = 1000, learning_rate = 0.7)
	adaboost.fit(X = X_train, y = y_train)
	ada_train.append(quadratic_weighted_kappa(adaboost.predict(X = X_train), y_train))
	ada_test.append(quadratic_weighted_kappa(adaboost.predict(X = X_test), y_test))
	print 'end'

trial_train = adaboost.staged_predict(X_train)
trial_test = adaboost.staged_predict(X_test)
result_train = []
result_test = []
for i in range(1000):
	if i%10 == 0:
		print i
	result_train.append(quadratic_weighted_kappa(trial_train.next(), y_train))
	result_test.append(quadratic_weighted_kappa(trial_test.next(),y_test))
pd.DataFrame({'n_estimator':range(1,1001), 'trian_result':result_train,'test_result': result_test}).to_csv('ADA.csv')
# gradient boosting method. 
