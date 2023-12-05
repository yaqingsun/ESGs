import numpy as np
import pandas as pd


class Population(object):
    def __init__(self, pop_size):
        # 导入数据
        # self.Matrix_a = np.array(pd.read_csv('data/MatrixA.csv', header=None), dtype='int16')  #方式组合-方式，0-1变量，30*20
        self.Matrix_b = np.array(pd.read_csv('data/MatrixB.csv', header=None), dtype='float16')  # 环境社会目标-方式，客观值，有正有负，15*20
        self.Matrix_b_sx = np.array(pd.read_csv('data/MatrixB_sx.csv', header=None), dtype='float16')  # 目标上限，客观值，15*1
        self.Matrix_b_xx = np.array(pd.read_csv('data/MatrixB_xx.csv', header=None), dtype='float16')  # 目标下限，客观值，15*1
        self.Matrix_b_cz =  np.array(pd.read_csv('data/MatrixB_cz.csv', header=None), dtype='float16')  # 目标上限-目标下限，客观值，15*1
        # self.Matrix_d = np.array(pd.read_csv('data/MatrixD.csv', header=None), dtype='float16')  # 治理目标-治理活动，[0,1],7*9
        # self.Matrix_e = np.array(pd.read_csv('data/MatrixE.csv', header=None), dtype='float16')  # 环境社会目标-治理活动，[0,1],15*9
        self.Matrix_f = np.array(pd.read_csv('data/MatrixF.csv', header=None), dtype='float16') # 环境社会目标-治理目标，[0,1],15*7
        self.Matrix_cost = np.array(pd.read_csv('data/Matrix_cost.csv', header=None), dtype='float16')  # 单个方式-节约的成本
        self.Matrix_w1 = np.array(pd.read_csv('data/Matrix_w1.csv', header=None), dtype='float16')  # 方式-弃渣利用，客观值，15*1
        self.Matrix_w2 = np.array(pd.read_csv('data/Matrix_w2.csv', header=None), dtype='float16')  # 方式-弃渣利用，客观值，15*1
        self.Matrix_w3 = np.array(pd.read_csv('data/Matrix_w3.csv', header=None), dtype='float16')  # 方式-弃渣利用，客观值，15*1
        self.Matrix_w4 = np.array(pd.read_csv('data/Matrix_w4.csv', header=None), dtype='float16')  # 方式-弃渣利用，客观值，15*1
        # 参数配置
        self.pop_size = pop_size  # 种群容量
        self.n_generations = 500  # 迭代代数
        self.mutation_rate = 0.8  # 变异率
        # 参数计算
        self.n_business = self.Matrix_f.shape[1]  # 治理目标数量 7
        self.n_capabilities = self.Matrix_b.shape[0]  # 需要支撑的环境社会目标数量 15
        self.dna_size = self.Matrix_b.shape[1]  # 处理利用废弃物的方式 20
        # 设置矩阵
        self.matrix_T = np.transpose(self.Matrix_b)
        self.matrix_C = self.Matrix_cost
        # 种群初始化
        self.dna = np.random.randint(2, size=(self.pop_size, self.dna_size))  # 生成种群的二进制编码
        self.history_dna = self.dna
        self.history_first = self.dna
        # 输出数据
        self.output = []

    def crossover_and_mutation(self, a, b):
        cross_points = np.random.randint(low=0, high=self.dna_size)  # 随机产生交叉的点
        child = a
        child[cross_points:] = b[cross_points:]
        if np.random.rand() < self.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, self.dna_size)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

        return child

    def mutation(self, dna):
        for i in range(len(dna)):
            mutation_dna_code = dna[i]
            mutate_point = np.random.randint(0, dna.shape[1])  # 随机产生一个实数，代表要变异基因的位置
            mutation_dna_code[mutate_point] = mutation_dna_code[mutate_point] ^ 1  # 将变异点的二进制为反转
        return dna

    def get_f1(self, dna):
        print('Shape of dna:', dna.shape)
        f1 = []
        matrix_c_1 = np.dot(self.Matrix_b, np.transpose(dna)) #环境目标-方式，客观值，15*pop-size(100)
        matrix_c_p_1 = np.array([])
        for i in range(dna.shape[0]):
            for j in range(matrix_c_1.shape[0]):
                matrix_c_p_1 = np.append(matrix_c_p_1, sum(self.Matrix_b[j, :] * dna[i, :]))
        matrix_c_p_1 = matrix_c_p_1.reshape((dna.shape[0], matrix_c_1.shape[0]))#环境目标-方式组合，客观值，pop-size(100)*15
        matrix_c_p_1_transposed = np.transpose(matrix_c_p_1)
        reshaped_matrix_b_xx = self.Matrix_b_xx.reshape(-1, 1)
        reshaped_matrix_b_cz = self.Matrix_b_cz.reshape(-1, 1)
        matrix_b_cz_expanded = np.tile(reshaped_matrix_b_cz, (1, dna.shape[0]))
        matrix_b_xx_expanded = np.tile(reshaped_matrix_b_xx, (1, dna.shape[0]))
        matrix_b_cz_expanded = np.tile(reshaped_matrix_b_cz, (1, dna.shape[0]))

        # print('Shape of matrix_c_p_1_transposed:', matrix_c_p_1_transposed.shape)
        # print('Shape of matrix_b_xx_expanded:', matrix_b_xx_expanded.shape)
        # print('Shape of matrix_b_cz_expanded:', matrix_b_cz_expanded.shape)
        matrix_c_p = np.divide((matrix_c_p_1_transposed - matrix_b_xx_expanded), matrix_b_cz_expanded) #环境社会-方式组合，pop—size*15，[0,1]
        matrix_c = np.copy(matrix_c_p)
        matrix_c[matrix_c_p > 0] = 1
        matrix_c = np.clip(matrix_c, a_min=0, a_max=1)#环境社会-方式组合，pop—size*15，0-1变量
        matrix_g = np.dot(np.transpose(matrix_c), self.Matrix_f)
        # 方式组合-治理目标，pop-size*7，[0,1]
        requirement = np.dot(np.ones((1, self.Matrix_f.shape[0])), self.Matrix_f)
        # 两者比较
        compare = matrix_g - requirement
        matrix_g_normalize = np.clip(compare, a_min=-1, a_max=0) + 1

        n_business_cover = np.sum(matrix_g_normalize, axis=1)
        f11 = n_business_cover / self.n_business

        # 计算f11因子: 使命任务数量覆盖因子
        matrix_c_1 = np.dot(self.Matrix_b, np.transpose(dna))

        # 计算f12因子: 业务能力实现因子
        matrix_c_p_1 = np.array([])
        for i in range(dna.shape[0]):
            for j in range(matrix_c_1.shape[0]):
                matrix_c_p_1 = np.append(matrix_c_p_1, sum(self.Matrix_b[j, :] * dna[i, :]))
        matrix_c_p_1 = matrix_c_p_1.reshape((dna.shape[0], matrix_c_1.shape[0]))
        column_array = self.Matrix_b_sx - self.Matrix_b_xx
        matrix_c_p = np.transpose(
            (np.transpose(matrix_c_p_1) - matrix_b_xx_expanded) / matrix_b_cz_expanded
        )
        matrix_g_p = np.array([])
        for i in range(dna.shape[0]):
            for j in range(matrix_g.shape[1]):
                support = self.Matrix_f[:, j] * matrix_c_p[i, :]
                # min_support = sum(filter(lambda item: item > 0, support.tolist()),default=0)
                min_support = sum(item if item > 0 else 0 for item in support.tolist())/15###############
                matrix_g_p = np.append(matrix_g_p, min_support)
        matrix_g_p = matrix_g_p.reshape((dna.shape[0], matrix_g.shape[1]))
        matrix_g_p = np.clip(matrix_g_p, a_min=0, a_max=1)
        matrix_h = matrix_g_normalize * matrix_g_p
        f12 = np.sum(matrix_h, axis=1) / matrix_g.shape[1]

        w = [0.2, 0.8]
        for i in range(dna.shape[0]):
            f1_i = f11[i] * w[0] + f12[i] * w[1]  # 计算适应度
            f1.append(f1_i)
        f1 = np.clip(f1, a_min=0, a_max=1)
        f1 = np.array(f1, dtype='float16')

        matrix = [np.transpose(dna), self.Matrix_b, matrix_c, self.Matrix_f,
                  matrix_g_p]
        am = np.transpose(dna)
        np.set_printoptions(threshold = np.inf)
        print(am)
        print(f11)
        print(f12)
        print(f1)

        return f1, f11, f12, matrix

    def get_f2(self, dna):
        f2 = []
        costs = np.dot(dna, self.Matrix_cost)  # 计算当前种群中每个方式组合的成本
        for i in range(dna.shape[0]):
            f2_i = costs[i]
            f2_i = 1-f2_i[0]
            f2.append(f2_i)
        f2 = np.array(f2, dtype='float16')

        print(f2)
        return f2

    def get_f4(self, dna):
        f4 = []
        w1s = np.dot(dna, self.Matrix_w1)  # 计算方式组合中弃渣1总量的约束
        for i in range(dna.shape[0]):
            f4_i = w1s[i]
            f4_i = f4_i[0]
            f4.append(f4_i)
        f4 = np.array(f4, dtype='float16')
        return f4


    def get_f5(self, dna):
        f5 = []
        w2s = np.dot(dna, self.Matrix_w2)  # 计算方式组合中弃渣2总量的约束
        for i in range(dna.shape[0]):
            f5_i = w2s[i]
            f5_i = f5_i[0]
            f5.append(f5_i)
        f5 = np.array(f5, dtype='float16')
        return f5


    def get_f6(self, dna):
        f6 = []
        w3s = np.dot(dna, self.Matrix_w3)  # 计算方式组合中弃渣3总量的约束
        for i in range(dna.shape[0]):
            f6_i = w3s[i]
            f6_i = f6_i[0]
            f6.append(f6_i)
        f6 = np.array(f6, dtype='float16')
        return f6


    def get_f7(self, dna):
        f7 = []
        w4s = np.dot(dna, self.Matrix_w4)  # 计算方式组合中弃渣4总量的约束
        for i in range(dna.shape[0]):
            f7_i = w4s[i]
            f7_i = f7_i[0]
            f7.append(f7_i)
        f7 = np.array(f7, dtype='float16')
        return f7


def delete_dna4(self, max_f4, min_f4):
    f4 = self.get_f4(self.dna)
    delete_index = []
    for i in range(self.dna.shape[0]):
        if f4[i] > max_f4:
            delete_index.append(i)
        elif f4[i] < min_f4:
            delete_index.append(i)
    self.dna = np.delete(self.dna, delete_index, axis=0)


def delete_dna5(self, max_f5, min_f5):
    f5 = self.get_f5(self.dna)
    delete_index = []
    for i in range(self.dna.shape[0]):
        if f5[i] > max_f5:
            delete_index.append(i)
        elif f5[i] < min_f5:
            delete_index.append(i)
    self.dna = np.delete(self.dna, delete_index, axis=0)


def delete_dna(self, max_f6, min_f6):
    f6 = self.get_f6(self.dna)
    delete_index = []
    for i in range(self.dna.shape[0]):
        if f6[i] > max_f6:
            delete_index.append(i)
        elif f6[i] < min_f6:
            delete_index.append(i)
    self.dna = np.delete(self.dna, delete_index, axis=0)


def delete_dna(self, max_f7, min_f7):
    f7 = self.get_f7(self.dna)
    delete_index = []
    for i in range(self.dna.shape[0]):
        if f7[i] > max_f7:
            delete_index.append(i)
        elif f7[i] < min_f7:
            delete_index.append(i)
    self.dna = np.delete(self.dna, delete_index, axis=0)

def delete_dna(self, max_f2):
    f2 = self.get_f7(self.dna)
    delete_index = []
    for i in range(self.dna.shape[0]):
        if f2[i] > max_f2:
            delete_index.append(i)
    self.dna = np.delete(self.dna, delete_index, axis=0)


