import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data_2011 = pd.read_csv('./data/gt_2011.csv')
data_2012 = pd.read_csv('./data/gt_2012.csv')
data_2013 = pd.read_csv('./data/gt_2013.csv')
data_2014 = pd.read_csv('./data/gt_2014.csv')
data_2015 = pd.read_csv('./data/gt_2015.csv')


data = pd.concat([data_2011,data_2012,data_2013,data_2014,data_2015])

corr = []
for index in data.columns:
    corr.append(data.corrwith(data.loc[:,index],method='pearson').values)
corr = np.array(corr)
corr = corr[:-2,:-2]

plt.figure(figsize=(10,10))
ax = plt.axes()
for (i, j), z in np.ndenumerate(corr):
    plt.text(j, i, '%.2f'%(z), ha='center', va='center',weight='bold',size=15)

plt.title('Correlation matrix of input features')
c = plt.imshow(corr)
plt.colorbar(c)
majors = data.columns[:-2]
ax.tick_params(axis=u'both', which=u'both',length=0,size = 10,labelsize='large')

plt.xticks(np.arange(9), majors)  # Set text labels.
plt.yticks(np.arange(9), majors)  # Set text labels.
plt.savefig('./result/Correlation matrix.png')
plt.show()

# plt.figure(figsize=(10,2))
# ax = plt.axes()

# for i in range(9):
#     plt.text(i, 0, '%.2f'%(corr[7,i]), ha='center', va='center')

# plt.title('Correlation matrix of input features')
# c = plt.imshow(corr[7,:].reshape(1,-1))
# plt.colorbar(c)
# majors = data.columns[:-2]
# print(majors[7])
# ax.tick_params(axis=u'both', which=u'both',length=0)
# plt.xticks(np.arange(9), majors)  # Set text labels.
# plt.yticks(np.arange(1), [majors[7]])  # Set text labels.
# # plt.savefig('./result/Correlation matrix.png')
# plt.show()



