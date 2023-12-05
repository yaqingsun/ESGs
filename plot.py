import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def fig_config(x_min, x_max, y_min, y_max, xlabel, ylabel):
    figure = plt.figure(figsize=(12, 12), dpi=80)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': 28})
    plt.ylabel(ylabel, fontdict={'family': 'Times New Roman', 'size': 28})
    plt.yticks(fontproperties='Times New Roman', size=22)
    plt.xticks(fontproperties='Times New Roman', size=22)
    return figure


# 保存帕累托前沿面积
data_6 = np.array(pd.read_csv('data/surface-2023-10-21-1132.csv'), dtype='float32')
data_8 = np.array(pd.read_csv('data/surface-2023-10-21-1134.csv'), dtype='float32')
data_10 = np.array(pd.read_csv('data/surface-2023-10-21-1135.csv'), dtype='float32')
data_12 = np.array(pd.read_csv('data/surface-2023-10-21-1145.csv'), dtype='float32')
data_14 = np.array(pd.read_csv('data/surface-2023-10-21-1146.csv'), dtype='float32')
data_16 = np.array(pd.read_csv('data/surface-2023-10-21-1147.csv'), dtype='float32')
data_18 = np.array(pd.read_csv('data/surface-2023-10-21-1148.csv.'), dtype='float32')
data_19 = np.array(pd.read_csv('data/surface-2023-10-21-1152.csv'), dtype='float32')


fig = fig_config(1, 100, 0, 5, 'generations', 'surface of pareto frontier')

plt.plot(data_6[:, 0], data_6[:, 1], linestyle=':', marker='', color='#71ae46', label='capacity 6')
plt.plot(data_8[:, 0], data_8[:, 1], linestyle=':', marker='', color='#c4cc38', label='capacity 8')
plt.plot(data_10[:, 0], data_10[:, 1], linestyle=':', marker='', color='#eab026', label='capacity 10')
plt.plot(data_12[:, 0], data_12[:, 1], linestyle=':', marker='', color='#d85d2a', label='capacity 12')
plt.plot(data_14[:, 0], data_14[:, 1], linestyle=':', marker='', color='#ac2026', label='capacity 14')
plt.plot(data_16[:, 0], data_16[:, 1], linestyle=':', marker='', color='#96b744', label='capacity 16')
plt.plot(data_18[:, 0], data_18[:, 1], linestyle=':', marker='', color='black', label='capacity 18')
plt.plot(data_19[:, 0], data_19[:, 1], linestyle='-', marker='', color='black', label='capacity 19')
plt.legend(loc='lower right', prop={'size': 18})
plt.show()


fig.savefig('data/plot1.eps', dpi=300, format='eps')
