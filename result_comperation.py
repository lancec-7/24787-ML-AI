import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

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

    # load the data
    trainX, train_CO, train_NOX= load_train()
    testX, test_CO, test_NOX = load_test()

    trainX = normalize_Logistic(trainX)
    testX = normalize_Logistic(testX)

    trainX = np.delete(trainX,(4,5,8),1)
    testX = np.delete(testX,(4,5,8),1)
    
    # train CO
    model_CO = LinearRegression()
    model_CO.fit(trainX,train_CO)

    predict_CO = model_CO.predict(testX)
    MAE_CO = np.mean(abs(predict_CO - test_CO))


    print('MAE of CO:%.2f'%(MAE_CO))
    print('mean of predict:',np.mean(predict_CO))
    print('mean of data:',np.mean(test_CO))
    print('*' * 50)


    plt.figure(figsize=(5,5))
    plt.scatter(test_CO,predict_CO,s=0.02)
    plt.axis('equal')
    plt.xlim([-2, 15])
    plt.ylim([-2, 15])
    plt.xlabel('true CO')
    plt.ylabel('predict CO')
    plt.savefig('./result/result_compare_CO.png')
    plt.show()
    

    # load the data
    trainX, train_CO, train_NOX= load_train()
    testX, test_CO, test_NOX = load_test()

    trainX = normalize_MinMax(trainX)
    testX = normalize_MinMax(testX)

    # train NOX
    model_NOX = LinearRegression()
    model_NOX.fit(trainX,train_NOX)

    predict_NOX = model_NOX.predict(testX)
    MAE_NOX = np.mean(abs(predict_NOX - test_NOX))

    print('MAE of NOX:%.2f'%(MAE_NOX))
    print('mean of predict:',np.mean(predict_NOX))
    print('mean of data:',np.mean(test_NOX))

    plt.figure(figsize=(5,5))

    plt.scatter(test_NOX,predict_NOX,s=0.02)
    plt.axis('equal')
    plt.xlim([40, 100])
    plt.ylim([40, 100])
    plt.xlabel('true NOX')
    plt.ylabel('predict NOX')
    plt.savefig('./result/result_compare_NOX.png')
    plt.show()
    

if __name__ == '__main__':
    main()