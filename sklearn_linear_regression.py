from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def load_data():
    train_data_2011 = pd.read_csv('./data/gt_2011.csv').values
    train_data_2012 = pd.read_csv('./data/gt_2012.csv').values
    train_data_2013 = pd.read_csv('./data/gt_2013.csv').values
    train_data_2014 = pd.read_csv('./data/gt_2014.csv').values


    trainX = np.vstack((train_data_2011[:,:-2],train_data_2012[:,:-2],train_data_2013[:,:-2],train_data_2014[:,:-2]))
    train_CO = np.hstack((train_data_2011[:,-2],train_data_2012[:,-2],train_data_2013[:,-2],train_data_2014[:,-2]))
    train_NOX = np.hstack((train_data_2011[:,-1],train_data_2012[:,-1],train_data_2013[:,-1],train_data_2014[:,-1]))

    return trainX, train_CO, train_NOX

def load_test():
    test_data_2015 = pd.read_csv('./data/gt_2015.csv').values
    testX = test_data_2015[:,:-2]
    test_CO = test_data_2015[:,-2]
    test_NOX = test_data_2015[:,-1]
    return testX, test_CO, test_NOX

def main():

    # load the data
    trainX, train_CO, train_NOX = load_data()
    testX, test_CO, test_NOX = load_test()

    # train CO
    model_CO = LinearRegression()
    model_CO.fit(trainX,train_CO)

    predict_CO = model_CO.predict(testX)
    MSE_CO = np.mean((predict_CO - test_CO)**2)

    print('MSE of CO:',MSE_CO)
    print('mean of predict:',np.mean(predict_CO))
    print('mean of data:',np.mean(test_CO))
    print('*' * 50)
    # train NOX
    model_NOX = LinearRegression()
    model_NOX.fit(trainX,train_NOX)

    predict_NOX = model_NOX.predict(testX)
    MSE_NOX = np.mean((predict_NOX - test_NOX)**2)

    print('MSE of NOX:',MSE_NOX)
    print('mean of predict:',np.mean(predict_NOX))
    print('mean of data:',np.mean(test_NOX))



if __name__ == '__main__':
    main()