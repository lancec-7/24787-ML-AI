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

def plot_feature_label_relation(data,feature_name):
    
    for a in range(-2,0):
        fig, axs = plt.subplots(3,3,figsize=(12,8))
        fig.suptitle('Relation between Feature and %s subplots'%(feature_name[a]))
        for i in range(3):
            for j in range(3):
                axs[i,j].scatter(data[:,3*i+j],data[:,a],s=0.1)
                axs[i,j].set(xlabel=feature_name[3*i+j], ylabel=feature_name[a])
        plt.tight_layout()
        plt.savefig('./result/Feature_%s_relation_subplots.png'%(feature_name[a]))
        plt.show()

if __name__ == '__main__':
    data,feature_name = load_data()
    plot_feature_label_relation(data,feature_name)
