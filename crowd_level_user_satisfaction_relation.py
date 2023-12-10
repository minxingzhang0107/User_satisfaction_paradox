import numpy
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import sem
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import gmean,hmean
import random
import math
sns.set_theme()
print("UR   CL")
dataframe=pd.read_excel('../data/526.xlsx',index_col=0)
crowd_level_list = []
ax = plt.axes()
for i in range (5):
    data = dataframe[dataframe['User_satisfaction'].isin([i+1])]
    crowd_level = data['crowd_level']
    crowd_level_list.append(crowd_level.mean())
    print(i+1,crowd_level.mean())
    sns.kdeplot(crowd_level, ax=ax, label="User rating = "+str(i+1))
plt.legend()
plt.savefig('../data/Ur_CL_Dis.png')
plt.show()
sns.barplot(y=crowd_level_list, x=list(range(1, 6)))
# plt.ylim(0.35, 0.5)
plt.savefig('../data/Ur_CL.png')
plt.show()
crowd_status_name = list(set(dataframe['crowd_status']))
US_list = []
for i in range (len(crowd_status_name)):
    data = dataframe[dataframe['crowd_status']==crowd_status_name[i]]
    User_satisfaction = data['User_satisfaction']
    US_list.append(User_satisfaction.mean())
    print(crowd_status_name[i],User_satisfaction.mean())
    sns.kdeplot(User_satisfaction, ax=ax, label='crowd_status'+crowd_status_name[i])
plt.savefig('../data/CS_US_Dis.png')
plt.show()
sns.barplot(y=US_list,x=crowd_status_name)
# plt.ylim(3.5,5.0)
plt.savefig('../data/CS_US.png')
plt.show()
crowd_level = dataframe['crowd_level']
rating = dataframe['User_satisfaction']
my_rho = np.corrcoef(np.array(crowd_level), np.array(rating))
print(my_rho)