import Model
import numpy as np
import pandas as pd
from pandas.core.common import flatten
import matplotlib.pyplot as plt
import utils
import time
#定义其他变量


def first_rank_preserved(dna_index, dna): #输入的两个参数
    """
    只保留第一分层的基因
    :param dna_index:
    :param dna:
    :return: 第一分层基因
    """
    # 将索引展平
    first_rank_dna_index = [list(flatten(dna_index[0]))] #将dna index的第一个元素展平得到的一维列表
    first_rank_dna_code = dna[tuple(first_rank_dna_index)] #从dna中提取的第一分层基因
    return first_rank_dna_code
def elitism_preserved(dna_index, dna, dna_capacity): #输入的三个参数
    """
    精英保留策略
    :param dna_index:
    :param dna: 输入种群
    :param dna_capacity: 保留基因的数量, 最终保留的基因数量可能超过此数值, 具体数量取决于第一分层的基因数量
    :return: 精英种群, 第一分层基因数量
    """
    # 获取第一分层基因的数量
    n_first_rank_dna = len(dna_index[0]) #获取的第一分层基因的数量
    first_rank_dna_index = dna_index[0] #获取第一分层基因的索引

    # 剩下基因的数目
    if n_first_rank_dna > dna_capacity:#第一分层大于需要保存的数目
        n_left_dna = M_0 #初始时基因数目
    elif n_first_rank_dna * 2 > dna_capacity:
        n_left_dna = n_first_rank_dna #第一分层基因的数量
    else:
        n_left_dna = dna_capacity - n_first_rank_dna #容量-第一分层数量

    # 保留其余分层 按三个指标
    elite_dna_index = []
    rank_num = 1 #需要保留的最高层级（初始为1
    n_dna = 0  # 记录剩下分层已保存的分层的基因数量 初始为0
    # 从第二分层开始，计算到哪一层将超出上限，将层数保存在rank_num，将基因数量保存在n_dna
    for i in range(1, len(dna_index)):
        n_dna = n_dna + len(dna_index[i])  # 下一分层引入后的基因数量
        if n_dna > n_left_dna:  # 如果引入下一分层后基因数量超出上限
            n_dna = n_dna - len(dna_index[i])  # 之前分层的基因数量
            rank_num = i  # 当前的层数
            break
    # 将前几个分层的基因保存
    for i in range(1, rank_num):
        elite_dna_index.append(dna_index[i])

    # 计算最后一层基因的拥挤度
    index = dna_index[rank_num] #最后一层索引
    f1 = pop.get_f1(pop.dna)[0]
    f2 = pop.get_f2(pop.dna)
    dist = utils.crowding_distance(f1[:], f2[:], index)#计算每个基因拥挤度
    # 找出拥挤度最高的基因
    last_list = []
    for i in range(0, n_left_dna - n_dna):
        # 找出拥挤度最高的基因
        index_needed = utils.index_of(max(dist), dist) #拥挤度最高
        last_list.append(dna_index[rank_num][index_needed])
        dist[utils.index_of(max(dist), dist)] = -1 #将该拥挤度设置为-1
    elite_dna_index.append(last_list)

    # 记录保存下来的基因分层情况
    preserved_layers = [first_rank_dna_index] + elite_dna_index #第一层基因索引和拥挤度最高的基因索引
    # 统计各分层基因数量
    dna_num_in_layers = []
    for layer in preserved_layers:
        dna_num_in_layers = np.append(dna_num_in_layers, len(layer))
    # 层数最大设置为20，不足20的话补0
    if len(dna_num_in_layers) < 20:
        dna_num_in_layers = np.append(dna_num_in_layers, np.zeros((20 - len(dna_num_in_layers)))) #各分层基因的数量信息

    # 将索引展平
    first_rank_dna_index = list(flatten(first_rank_dna_index))
    elite_dna_index = list(flatten(elite_dna_index))

    # 将优秀基因保存下来
    elite_dna_code = np.vstack((dna[first_rank_dna_index], dna[elite_dna_index]))#将第一层基因和拥挤度最高的基因合成一个矩阵

    return elite_dna_code, len(first_rank_dna_index), dna_num_in_layers


def fig_config(x_min, x_max, y_min, y_max, xlabel, ylabel):
    figure = plt.figure(figsize=(12, 12), dpi=80)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': 28})
    plt.ylabel(ylabel, fontdict={'family': 'Times New Roman', 'size': 28})
    plt.yticks(fontproperties='Times New Roman', size=22)
    plt.xticks(fontproperties='Times New Roman', size=22)
    return figure#绘图


# 变异策略 这个函数用于实现变异策略，通过随机生成长度为dna len的数组来获取变异的基因，然后通过交叉和变异的方式来扩充种群，最后计算非支配解来获取最优解。
def gen_mutation_index(dna_len):
    return np.random.randint(dna_len, size=(1, dna_len))[0] #在整数范围内随机生成长度为dna len的数组


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) #对数据进行激活，将随机生成的数据保证到（0,1）之间


if __name__ == '__main__':
    n_first_rank = 0
    surface = 0 #种群表面变化
    sorted_plot_data = [] #记录种群适应度排序
    first_rank_dna = []
    M_0 = 24# M_0
    surface_iter = []  #储存每代种群的surface值
    pop = Model.Population(M_0)#对M0进行初始化 pop受M0的影响

    fig = fig_config(0.2, 1,-0.1, 1.1, 'fitness2', 'fitness1') #配置图像的参数
    n_in_all_layers_list = np.zeros(20)  # 记录每次保存的分层情况 长度为20 此时全为0
    # 开始遗传迭代
    for gen_no in range(pop.n_generations):  # 迭代n代
        # 随机选择一部分上代基因复制
        mutation_index = gen_mutation_index(pop.dna.shape[0]) #随机生成（1，dna_len）的数组 行数为1 列数取决于DNA序列
        mutation_dna = pop.mutation(np.copy(pop.dna[mutation_index])) #储存变异后的dna序列

        # 通过交叉和变异将基因进行扩充
        child_dna = np.array([], dtype=int)
        for i in range(pop.dna.shape[0]):
            father_dna = np.copy(pop.dna[i]) #选择父基因
            mother_index = np.random.randint(0, pop.dna.shape[0] - 1)
            mother_dna = np.copy(pop.dna[mother_index]) #选择母基因
            child_dna = np.append(child_dna, pop.crossover_and_mutation(father_dna, mother_dna)) #产生的子基因 一维数组 列数不能确定

        # 扩充种群 会得到多种堆砌结果
        pop.dna = np.vstack((pop.dna, child_dna.reshape((pop.dna.shape[0], pop.dna_size)))) #子基因转化为与pop.dna形状相同的矩阵 堆砌
        pop.dna = np.vstack((pop.dna, mutation_dna)) #变异基因垂直堆砌在pop.dna的下方
        #pop.dna代表种群的基因组成，每行表示一个个体的基因，每列表示一个基因的取值 行数增加


        # 计算非支配解
        fit1 = pop.get_f1(pop.dna)[0]
        fit2 = pop.get_f2(pop.dna)
        non_dominated_dna_index = utils.fast_non_dominated_sort(fit1, fit2) #进行排序，多支配在后，非支配在前（适应度）

        # 删除重复解 除第一分层与精英个体其他都会被删除
        _ = utils.delete_same_solution(non_dominated_dna_index, pop.dna, 1) #删除重复解
        first_rank_dna = first_rank_preserved(non_dominated_dna_index, pop.dna) #保留第一分层
        pop.dna, n_first_rank, n_in_all_layers = elitism_preserved(non_dominated_dna_index, pop.dna, M_0) #保留精英个体
        n_in_all_layers_list = np.vstack((n_in_all_layers_list, n_in_all_layers)) #将n_in_all_layers堆砌至n_in_all_layers_list
        #popdna保留精英个体的种群 n first rank处理后剩余第一分层的基因数 n in all layers每个分层中剩余的个体的数量 n_in_all_layers_list每一行记录一个代的每一层中剩余的个体数
        # 绘制帕累托前沿
        sorted_plot_data = utils.sort_plot_data(
            np.vstack((np.array(pop.get_f2(first_rank_dna)), np.array(pop.get_f1(first_rank_dna)[0])))) #对first rank dna 进行排序
        if gen_no == 0:
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle=':', marker='', color=[0, 0, 0, 0.3])
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle='', marker='o', color='blue', label='Generation 1')
            plt.legend(loc='upper left', prop={'size': 18})
            # plt.show()
        elif gen_no == 4:
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle=':', marker='', color=[0, 0, 0, 0.3])
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle='', marker='o', color='orange',
                 label='Generation 5')
            plt.legend(loc='upper left', prop={'size': 18})
            # plt.show()

        elif gen_no == 9:
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle=':', marker='', color=[0, 0, 0, 0.3])
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle='', marker='o', color='brown',
                 label='Generation 10')
            plt.legend(loc='upper left', prop={'size': 18})
            # plt.show()
        # elif gen_no == 19:
        #     plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle=':', marker='', color=[0, 0, 0, 0.3])
        #     plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle='', marker='o', color='olivedrab',
        #          label='Generation 20')
        #     plt.legend(loc='upper left', prop={'size': 18})
        #     # plt.show()

        elif gen_no == 99:
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle=':', marker='', color=[0, 0, 0, 0.3])
            plt.plot(sorted_plot_data[0], sorted_plot_data[1], linestyle='', marker='o', color='purple',
                 label='Generation 100')
            plt.legend(loc='upper left', prop={'size': 18})
            plt.show()
            break


    # plt.legend(loc='upper left', prop={'size': 12})
    #     plt.show()


        # 计算当前帕累托前沿的面积
        surface = utils.cal_first_rank_surface(sorted_plot_data, 0)
        print('Iter{}: [{}/{}] surface: {}'.format(gen_no, n_first_rank, pop.dna.shape[0], surface))
        surface_iter.append(surface)

    # 绘制帕累托前沿
    plt.show()

    # 转为十六进制
    first_rank_code = []
    for dna_i in first_rank_dna:
        first_rank_code.append([hex(int("".join(map(lambda x:str(x), list(dna_i))), 2))])

    # 保存帕累托前沿结果
    first_rank_output = []
    n_projects = sum(np.transpose(first_rank_dna))
    fit1, fit11, fit12, matrix = pop.get_f1(first_rank_dna)
    cost = 1 - pop.get_f2(first_rank_dna)
    for i in range(len(first_rank_dna)):
        first_rank_output.append([fit11[i], fit12[i], cost[i], n_projects[i]])

    # 保存结果
    a1 = np.array(first_rank_code)
    a2 = np.array(first_rank_output)
    a3 = np.array(first_rank_dna)
    df = pd.DataFrame(np.hstack((np.hstack((a1, a2)), a3)))
    df.to_csv('data/output-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')

    df2 = pd.DataFrame(np.array(surface_iter))
    df2.to_csv('data/surface-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')

    df3 = pd.DataFrame(n_in_all_layers_list)
    df3.to_csv('data/dna_in_layers_list-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')

    mat_a = pd.DataFrame(matrix[0])
    mat_b = pd.DataFrame(matrix[1])
    mat_c = pd.DataFrame(matrix[2])
    mat_d = pd.DataFrame(matrix[3])
    mat_e = pd.DataFrame(matrix[4])
    mat_f = pd.DataFrame(matrix[5])
    mat_g = pd.DataFrame(matrix[6])
    mat_a.to_csv('data/matrix-a-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')
    mat_b.to_csv('data/matrix-b-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')
    mat_c.to_csv('data/matrix-c-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')
    mat_d.to_csv('data/matrix-d-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')
    mat_e.to_csv('data/matrix-e-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')
    mat_f.to_csv('data/matrix-f-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')
    mat_g.to_csv('data/matrix-g-' + time.strftime("%Y-%m-%d-%H%M") + '.csv')

    # 保存帕累托前沿
    fig.savefig('data/scatter-' + time.strftime("%Y-%m-%d-%H%M") + '.eps', dpi=300, format='eps')
    # 保存种群数量
    fig2 = fig_config(1, 50, 0, 80, 'generations', 'dna_num')
    pal = ["#71ae46", "#96b744", "#c4cc38", "#ebe12a", "#eab026", "#e3852b", "#d85d2a"]

    plt.stackplot(np.arange(101),
                  n_in_all_layers_list[:, 0],
                  n_in_all_layers_list[:, 1],
                  n_in_all_layers_list[:, 2],
                  n_in_all_layers_list[:, 3],
                  n_in_all_layers_list[:, 4],
                  n_in_all_layers_list[:, 5],
                  n_in_all_layers_list[:, 6],
                  labels=['1st rank', '2st rank', '3st rank', '4st rank', '5st rank', '6st rank', '7st rank'],
                  colors=pal)
    plt.legend(loc='upper left', prop={'size': 18})
    plt.show()
    fig2.savefig('data/dna-num-' + time.strftime("%Y-%m-%d-%H%M") + '.eps', dpi=300, format='eps')

