import pandas as pd 
import numpy as np 
from sklearn import cross_validation
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import accuracy_score
# setting the project directory
import os 
os.chdir('C:/Users/JianfengYan/Desktop/resorce one/internship')
from quadractic_weighted_kappa import *


#  function for create dummy variables for discrete data 
#  filling continuous data with mean value. 
#  for discrete variable, create new variable, 0 for non-missing, 1 for missing. 

def columns_manipulation(data, col, data_type = 'nominal'):
    if data_type == 'continuous':
        data[col] = data[col].fillna(data[col].mean())
        return
    data[col] = data[col].fillna('cindy')
    G = pd.get_dummies(data[col].apply(str))
    for i in G.columns: 
        name = col + '_' + i
        data[name] = G[i].astype(int)
    del data[col]

""" 

      function manipulate_data is for processing the data 
      parameter: 
            nominal_col: columns of nominal variables, create dummy variable for it. 
            conti_col: replacing missigng value with means 
            discrete_col: replacing missing value with means rounded to int 

"""


def manipulate_data(data, nominal_col = None, conti_col = 'cindy', discrete_col = None):
    if (nominal_col == None) & (conti_col == 'cindy') & (discrete_col == None):
        print 'nothing changed'
        return data
    if nominal_col != None:
        nominal_col = nominal_col.split(',')
        nominal_col = [i.strip() for i in nominal_col]
        for col in nominal_col:
            print col + ' has been processed'
            columns_manipulation(data, col)
    if conti_col != 'cindy':
        print "cindy's birthday is on the day before Christmas !"
        conti_col = conti_col.split(',')
        conti_col = [i.strip() for i in conti_col]
        for col in conti_col:
            print col + ' has been processed'
            columns_manipulation(data, col, 'continuous')
    if discrete_col != None: 
        discrete_col = discrete_col.split(',')
        discrete_col = [i.strip() for i in discrete_col]
        for i in discrete_col:
            data[i] = data[i].fillna(data[i].mean()).astype(np.int32)
    return data 


"""
      read train and test data
      pre-processing it for modeling 
      output X_train, X_test, y_train

"""
if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    y_train = train_data['Response']
    del train_data['Response']
    Id_test = test_data.Id
    data = pd.concat([train_data,test_data])

    # manipulate the nominal data  
    nominal_col = 'Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41'
    
    # manipulate the continuous data
    conti_col = 'Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5'

    # manipulate the discrete data. 
    discrete_col = 'Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32'

    data = manipulate_data(data, nominal_col, conti_col, discrete_col)

    # change the data type to make it less memory intensive
    for i in range(1,49):
        data['Medical_Keyword_'+ str(i)] = data['Medical_Keyword_'+ str(i)].astype(np.int16)

    # preparing the data for modeling  
    X_train = data.iloc[range(train_data.shape[0]),:]

    X_test = data.iloc[range((train_data.shape[0]),data.shape[0]),:]


