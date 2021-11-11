import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

def load_data():
    train_data_2011 = pd.read_csv('./data/gt_2011.csv').values
    train_data_2012 = pd.read_csv('./data/gt_2012.csv').values
    train_data_2013 = pd.read_csv('./data/gt_2013.csv').values
    train_data_2014 = pd.read_csv('./data/gt_2014.csv').values
    train_data_2015 = pd.read_csv('./data/gt_2015.csv').values

    trainX = np.vstack((train_data_2011[:,:-2],train_data_2012[:,:-2],train_data_2013[:,:-2],train_data_2014[:,:-2],train_data_2015[:,:-2]))
    train_CO = np.hstack((train_data_2011[:,-2],train_data_2012[:,-2],train_data_2013[:,-2],train_data_2014[:,-2],train_data_2015[:,-2]))
    train_NOX = np.hstack((train_data_2011[:,-1],train_data_2012[:,-1],train_data_2013[:,-1],train_data_2014[:,-1],train_data_2015[:,-1]))

    feature_name = [column for column in pd.read_csv('./data/gt_2011.csv')]
    return trainX, train_CO, train_NOX,feature_name


def plot_feature_distribution(data,feature_name):
    fig, axs = plt.subplots(3,3,figsize=(12,8))
    fig.suptitle('Feature distribution subplots')
    for i in range(3):
        for j in range(3):
            sns.histplot(data[:,3*i+j], kde=True, linewidth=0,ax=axs[i,j])
            axs[i,j].set(xlabel=feature_name[3*i+j], ylabel='Frequency')
    plt.tight_layout()
    plt.savefig('./result/Feature_distribution_subplots.png')
    plt.show()

if __name__ == '__main__':
    trainX, train_CO, train_NOX,feature_name = load_data()
    plot_feature_distribution(trainX,feature_name)
