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

    data = np.vstack((train_data_2011,train_data_2012,train_data_2013,train_data_2014,train_data_2015))
    feature_name = [column for column in pd.read_csv('./data/gt_2011.csv')]

    return data,feature_name

def normalize(data):
    # linear scaling

    mean = np.mean(data,axis=0)
    max = np.max(data,axis=0)
    min = np.min(data,axis=0)

    new_data = (data - mean)/(max-min)
    return new_data

def plot_comparation(data,feature_name):
    
    for a in range(-2,0):
        plt.figure(figsize=(12,8))
        for i in [4,7,8]:
            plt.scatter(data[:,i],data[:,a],s=0.1,label=feature_name[i])
        plt.legend()
        plt.tight_layout()
        plt.savefig('./result/Feature_%s_relation_comparation.png'%(feature_name[a]))
        plt.show()

if __name__ == '__main__':
    data,feature_name = load_data()
    data = normalize(data)
    plot_comparation(data,feature_name)
