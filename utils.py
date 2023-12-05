import numpy as np
import math


def sort_plot_data(plot_data):
    sorted_data = plot_data.T[np.lexsort(plot_data[::-1, :])].T
    return  sorted_data


def delete_same_solution(dna_index, input_dna, n_preserved):
    """
    寻找并删除相同解
    :param dna_index: 种群分层索引
    :param input_dna: 种群基因
    :param n_preserved: 相同的解最多保留几个
    :return:
    """
    n_delete = 0
    # 寻找各分层中的相同解
    for rank in range(0, len(dna_index)):
        delete_index = []  # 需要删除的index
        n_same_dna = [0 for i in range(0, len(dna_index[rank]))]  # 每个基因的相同解数量
        for p in range(0, len(dna_index[rank])):
            for q in range(p, len(dna_index[rank])):
                dna_index_a = dna_index[rank][p]
                dna_index_b = dna_index[rank][q]
                dna_diff = input_dna[dna_index_a] - input_dna[dna_index_b]
                n_non_zero = np.count_nonzero(dna_diff)
                if n_non_zero == 0:
                    n_same_dna[p] = n_same_dna[p] + 1
                if n_same_dna[p] > n_preserved:  # 如果相同解数量超过预设上限
                    if q not in delete_index:
                        delete_index.append(q)  # 将多余的相同解的index记录在delete_index中
                    n_same_dna[p] = n_preserved
        # 删除多余解的index
        if delete_index:
            dna_index[rank] = np.delete(np.array(dna_index[rank]), delete_index).tolist()
            n_delete = n_delete + 1
    return n_delete


def fast_non_dominated_sort(values1, values2):
    """
    非支配快速排序算法
    :param values1: 业务覆盖
    :param values2: 成本
    :return: 种群分层索引
    """
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        # 寻找第p个个体和其他个体的支配关系, 将第p个个体的sp和np初始化
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]

    return front


def index_of(a, a_list):
    for i in range(0, len(a_list)):
        if a_list[i] == a:
            return i
    return -1


def sort_by_values(list1, values):
    v = np.copy(values)  # 防止修改外部的values
    sorted_list = []
    while len(sorted_list) != len(list1):
        # 当结果长度不等于初始长度时，继续循环
        if index_of(min(v), v) in list1:
            # 标定值中最小值在目标列表中时
            sorted_list.append(index_of(min(v), v))
        #     将标定值的最小值的索引追加到结果列表后面
        v[index_of(min(v), v)] = math.inf
    #      将标定值的最小值置为无穷小,即删除原来的最小值,移向下一个
    return sorted_list


def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    # 初始化个体间的拥挤距离
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    # 基于目标函数1和目标函数2对已经划分好层级的种群排序
    distance[0] = 99
    distance[len(front) - 1] = 99
    for k in range(1, len(front) - 1):
        a = (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (max(values1) - min(values1) + 0.1)
        distance[k] = distance[k] + a
    for k in range(1, len(front) - 1):
        b = (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2) + 0.1)
        distance[k] = distance[k] + b
    return distance


def cal_first_rank_surface(sort_data, max_f2):
    surface = 0
    n = sort_data.shape[1] - 1
    for i in range(sort_data.shape[1]):
        a = sort_data[0][i] - max_f2
        b = sort_data[1][i]
        surface = surface + a * b
        max_f2 = sort_data[0][i]
    return surface
