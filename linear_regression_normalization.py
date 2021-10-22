import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


#  use the normalization
def feature_engineer(trainX):
    mu = np.mean(trainX,axis=0)
    sigma = np.sqrt(np.mean((trainX - mu)**2,axis=0))
    # print(sigma)
    # print(mu)
    trainX_new = trainX / sigma
    return trainX_new

class LinearRegression:

    def __init__(self):
        pass
        self.loss_list = []
    
    def compute_loss_and_gradient(self,trainX,trainY):
        Y_predict = trainX @ self.weights
        loss = np.mean((trainY-Y_predict)**2)
        # print(self.loss)
        self.loss_list.append(loss)
        self.gradient = 2 * trainX.T @ (Y_predict-trainY)/trainX.shape[0]
        # print(self.gradient)
        # self.loss = 1
        # self.gradient = 1
    
    def update(self,lr):
        self.weights -= lr * self.gradient

    def fit(self,trainX,trainY,lr=2e-5,max_iter=3000):
        pass

        trainX = np.hstack((np.ones((trainX.shape[0],1)),trainX))
        # print(trainX[0,:])
        self.feature_num = trainX.shape[1]
        # print(self.feature_num)
        self.weights = np.random.random(self.feature_num) 
        # print(self.weights)
    
        for _ in range(max_iter):
            self.compute_loss_and_gradient(trainX,trainY)
            self.update(lr)

        
        return self.weights

    
    def plot_loss(self):
        plt.plot(self.loss_list)
        plt.show()

    def predict(self,X):
        X = np.hstack((np.ones((X.shape[0],1)),X))
        Y_predict = X @ self.weights
        return Y_predict

    def score(self,X,Y):
        Y_predict = self.predict(X)
        loss = np.mean((Y_predict - Y)**2)
        return loss
        pass
    
    def save_weights(self,file):
        np.save(file,self.weights)

    def load_weight(self,file):
        self.weights = np.load(file)


def main():

    trainX, train_CO, train_NOX = load_data()

    # model_CO = LinearRegression()
    # weights_CO = model_CO.fit(trainX,train_CO)
    # print(weights_CO)
    # model_CO.plot_loss()
    # print(model_CO.score(trainX,train_CO))
    # model_CO.save_weights('./result/weights_CO.npy')

    # model_NOX = LinearRegression()
    # weights_NOX = model_NOX.fit(trainX,train_NOX)
    # print(weights_NOX)
    # # model_NOX.plot_loss()
    # print(model_NOX.score(trainX,train_NOX))
    # model_NOX.save_weights('./result/weights_NOX.npy')

    # print(model_NOX.predict(trainX[0:3,:]))
    testX, test_CO, test_NOX = load_test()
    


    trainX_new = feature_engineer(trainX)
    testX_new = feature_engineer(testX)
    testX = np.hstack((np.ones((testX.shape[0],1)),testX))


    # print(np.max(trainX_new,axis=0))
    model_NOX = LinearRegression()
    # weights_NOX = model_NOX.fit(trainX_new,train_NOX)
    # print(weights_NOX)
    # model_NOX.plot_loss()

    
    # model_NOX.save_weights('./result/weights_NOX.npy')
    model_NOX.load_weight('./result/weights_NOX.npy')
    print(model_NOX.score(trainX_new,train_NOX))
    print(model_NOX.score(testX_new,test_NOX))
    print(model_NOX.predict(testX_new[0:3,:]))




if __name__ == '__main__':
    main()