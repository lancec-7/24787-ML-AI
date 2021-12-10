from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import time

def load_train():
    train_data_2011 = pd.read_csv('./data/gt_2011.csv').values
    train_data_2012 = pd.read_csv('./data/gt_2012.csv').values
    train_data_2013 = pd.read_csv('./data/gt_2013.csv').values

    trainX = np.vstack((train_data_2011[:,:-2],train_data_2012[:,:-2],train_data_2013[:,:-2]))
    train_CO = np.hstack((train_data_2011[:,-2],train_data_2012[:,-2],train_data_2013[:,-2]))
    train_NOX = np.hstack((train_data_2011[:,-1],train_data_2012[:,-1],train_data_2013[:,-1]))

    return trainX, train_CO, train_NOX

def load_validation():
    validate_data_2014 = pd.read_csv('./data/gt_2014.csv').values
    validateX = validate_data_2014[:,:-2]
    validate_CO = validate_data_2014[:,-2]
    validate_NOX = validate_data_2014[:,-1]

    return validateX, validate_CO, validate_NOX

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
    start = time.time()


    # load the data
    trainX, train_CO, train_NOX= load_train()
    testX, test_CO, test_NOX = load_test()

    # trainX = normalize_MinMax(trainX)
    # testX = normalize_MinMax(testX)

    # trainX = normalize_Zscore(trainX)
    # testX = normalize_Zscore(testX)

    # trainX = normalize_Logistic(trainX)
    # testX = normalize_Logistic(testX)

    poly = PolynomialFeatures(2)
    trainX = poly.fit_transform(trainX)
    testX = poly.fit_transform(testX)

    # trainX = np.delete(trainX,(4,5,8),1)
    # testX = np.delete(testX,(4,5,8),1)

    # train CO
    model_CO = GradientBoostingRegressor()
    model_CO.fit(trainX,train_CO)

    predict_CO = model_CO.predict(testX)
    MAE_CO = np.mean(abs(predict_CO - test_CO))

    # # train NOX
    # model_NOX = GradientBoostingRegressor()
    # model_NOX.fit(trainX,train_NOX)

    # predict_NOX = model_NOX.predict(testX)
    # MAE_NOX = np.mean(abs(predict_NOX - test_NOX))

    stop = time.time()

    running_time = stop - start

    print('MAE of CO:%.2f'%(MAE_CO))
    print('mean of predict:',np.mean(predict_CO))
    print('mean of data:',np.mean(test_CO))
    print('*' * 50)

    # print('MSE of NOX:%.2f'%(MAE_NOX))
    # print('mean of predict:',np.mean(predict_NOX))
    # print('mean of data:',np.mean(test_NOX))
    # print('*' * 50)

    print('runtime: %.2fs'%(running_time))


if __name__ == '__main__':
    main()