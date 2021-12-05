from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_train():
    train_data_2011 = pd.read_csv('./data/gt_2011.csv').values
    train_data_2012 = pd.read_csv('./data/gt_2012.csv').values
    train_data_2013 = pd.read_csv('./data/gt_2013.csv').values

    X = np.vstack((train_data_2011[:,:-2],train_data_2012[:,:-2],train_data_2013[:,:-2]))
    CO = np.hstack((train_data_2011[:,-2],train_data_2012[:,-2],train_data_2013[:,-2]))
    NOX = np.hstack((train_data_2011[:,-1],train_data_2012[:,-1],train_data_2013[:,-1]))

    return X, CO, NOX

def load_test():
    test_data_2015 = pd.read_csv('./data/gt_2015.csv').values
    testX = test_data_2015[:,:-2]
    test_CO = test_data_2015[:,-2]
    test_NOX = test_data_2015[:,-1]

    return testX, test_CO, test_NOX

def normalize_MinMax(x):
    MAX = np.max(x,axis=0)
    MIN = np.min(x,axis=0)
    return (x-MIN)/(MAX - MIN)

def normalize_Zscore(x):
    MEAN = np.mean(x,axis=0)
    std = np.sqrt(np.mean((x - MEAN)**2,axis=0))
    return (x - MEAN)/std

def normalize_Logistic(x):
    return 1/(1+np.exp(-x))

def main():

    trainX, train_CO, train_NOX = load_train()
    testX, test_CO, test_NOX = load_test()

    # trainX = normalize_MinMax(trainX)
    # testX = normalize_MinMax(testX)

    # trainX = normalize_Zscore(trainX)
    # testX = normalize_Zscore(testX)

    # trainX = normalize_Logistic(trainX)
    # testX = normalize_Logistic(testX)

    trainX = np.delete(trainX,(4,5,8),1)
    testX = np.delete(testX,(4,5,8),1)

    randomforest = RandomForestRegressor()
    randomforest.fit(trainX,train_CO)
    CO_predict = randomforest.predict(testX)
    MAE_CO = np.mean(abs(CO_predict - test_CO))

    randomforest = RandomForestRegressor()
    randomforest.fit(trainX,train_NOX)
    NOX_predict = randomforest.predict(testX)
    MAE_NOX = np.mean(abs(NOX_predict - test_NOX))

    print('MAE of CO:',np.mean(MAE_CO))
    print('*' * 50)
    print('MAE of NOX:',np.mean(MAE_NOX))

if __name__ == '__main__':
    main()