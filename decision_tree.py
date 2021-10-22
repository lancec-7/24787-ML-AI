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

    # print(trainX.shape)
    # print(train_CO.shape)
    # print(train_NOX[0])
    return trainX, train_CO, train_NOX

class DecisionTree:
    def __init__(self,trainX,train_CO,train_NOX,max_depth):
        self.trainX = trainX
        self.train_CO = train_CO
        self.train_NOX = train_NOX

        

def main():

    trainX, train_CO, train_NOX = load_data()


if __name__ == '__main__':
    main()
