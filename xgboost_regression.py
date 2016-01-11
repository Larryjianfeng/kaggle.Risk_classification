import xgboost as xgb
from scipy.optimize import fmin_powell

def get_offset(preds, y):
	round_preds = np.round(preds)
 	off_sets = []
 	y_ = y.values
	for j in range(8):
		print j
		y_1 = preds[round_preds == j+1]
		y_2 = y_[round_preds == j+1]
		p = get_qwk(0,y_1, y_2)
		min_x = 0 
		for x in np.arange(-3,3,0.01):
			if get_qwk(x,y_1, y_2) > p:
				p = get_qwk(x,y_1,y_2)
				min_x = x 
		off_sets.append(np.round(min_x,3))
		print 'the class with risk ' + str(j + 1) + ' get the offset' + str(np.round(min_x,3))

	return off_sets


def get_qwk(x, y_1, y_2):
	y_1_new = np.clip(np.round(y_1+x),1,8)
	diff = np.round(y_1_new) - y_2 
	quratic_diff = diff**2
	quadratic_weighted_kappa_value = (len(y_1) - sum(quratic_diff)/49)/len(y_1)   
	return quadratic_weighted_kappa_value

def add_offset_to_prediction(preds, off_sets): 
	preds = np.clip(preds,1,8)
	round_preds = np.clip(np.round(preds),1,8)
	for j in range(8):
		offset = off_sets[j]
		preds[round_preds == j+1] = preds[round_preds == j+1] + offset
	return np.clip( np.round(preds),1,8 )

def get_params():

    """
    eta:  actually shrinks the feature weights afte each iteration of boosting,
     to make the boosting process more conservative
    objective: I tried linear regression and poisson regression, poission is better. 
    min_child_weight: minimum sum of instance weight needed in a child

    """
    params = {}
    params["objective"] = "count:poisson"     
    params["eta"] = 0.06
    params["min_child_weight"] = 80
    params["subsample"] = 0.85
    params["colsample_bytree"] = 0.30
    params["max_depth"] = 9
    plst = list(params.items())
    return plst




def make_trails():
	model = create_model()
	preds = model.predict(xgb.DMatrix(X_train.values))
	preds = np.clip(preds,1,8)
	quadratic_weighted_kappa(np.round(preds), y_train.values)

	off_sets = get_offset(preds, y_train)

	preds_train = model.predict(xgb.DMatrix(X_train.values))
	new_preds_train = add_offset_to_prediction(preds_train, off_sets )

	preds_test = model.predict(xgb.DMatrix(X_test.values))
	new_preds = add_offset_to_prediction(preds_test, off_sets)

	return [quadratic_weighted_kappa(np.clip(np.round(preds_test),1,8), y_test), quadratic_weighted_kappa(new_preds, y_test)]

def create_model():
	xgtrain = xgb.DMatrix(X_train.values, y_train.values)
	plst = get_params()
	xgb_num_rounds = 500
	model = xgb.train(plst, xgtrain, xgb_num_rounds)
	return model



eta_list = []
result_list = []
for eta in np.arange(0.05,0.15,0.01):
	print eta
	eta_list.append(eta)
	result_list.append(make_trails())
# using posssion regression with max length = 12, result 0.6148 vs 0.6464
# using posssion regression with max length = 12, result 0.6183 vs 0.6458
# using posssion regression with max length = 9, result 0.6159 vs 0.6478




# the result for submitting. 

model = create_model()
preds = model.predict(xgb.DMatrix(X_train.values))
preds = np.clip(preds,1,8)

off_sets = get_offset(preds, y_train)



preds_train = model.predict(xgb.DMatrix(X_train.values))
new_preds_train = add_offset_to_prediction(preds_train, off_sets )


# the new metrics with offsets
print quadratic_weighted_kappa(preds_train, y_train.values)
print quadratic_weighted_kappa(new_preds_train, y_train.values)


preds_test = model.predict(xgb.DMatrix(X_test.values))
new_preds = add_offset_to_prediction(preds_test, off_sets)


pd.DataFrame({'Id':X_test.Id.astype(int), 'Response':new_preds.astype(int)}).to_csv('submit.csv',index=False)