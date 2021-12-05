import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
   
model = torch.nn.Sequential(
        torch.nn.Linear(9, 512),
        torch.nn.Sigmoid(),
        # torch.nn.Linear(512, 512),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(512, 1),
    )

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

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

    trainX = normalize_Logistic(trainX)
    testX = normalize_Logistic(testX)

    # trainX = np.delete(trainX,(4,5,8),1)
    # testX = np.delete(testX,(4,5,8),1)


    my_loss = []
    for t in range(100):
        print(t,'*'*50)
        # train_set = TensorDataset(torch.Tensor(trainX), torch.Tensor(train_CO).type(torch.float))
        # train_loader = DataLoader(dataset=train_set, batch_size = 512, shuffle=True)
        
        running_loss = []

        # for i, data in enumerate(train_loader, 0):
        #     x_batch, y_batch = data
            
        #     optimizer.zero_grad()
        #     yhat = model(x_batch)
        #     loss = criterion(yhat, y_batch.view(-1,1))
        #     loss.backward()
        #     optimizer.step()
        #     running_loss.append(loss.item())
        # print(np.mean(running_loss))
        # my_loss.append(np.mean(running_loss))
 
        optimizer.zero_grad()
        yhat = model(torch.tensor(trainX).float())
        loss = criterion(yhat, torch.tensor(train_NOX).view(-1,1).float())
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        print(np.mean(running_loss))
        my_loss.append(np.mean(running_loss))

        
    # plt.plot(my_loss)
    # plt.show()

    # torch.save(model.state_dict(), './result/nn_model_CO')

    # model.load_state_dict(torch.load('./result/nn_model_CO'))
    model.eval()

    NOX_predict = model(torch.tensor(testX).float())

    MAE = torch.mean(abs(NOX_predict - torch.tensor(test_NOX).view(-1,1).float()))
    print('MAE of CO:',MAE.item())

if __name__ == '__main__':
    main()
