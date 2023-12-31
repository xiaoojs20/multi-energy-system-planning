import os
import numpy as np
import pandas as pd
from utils import *

class StdModel(object):
    def __init__(self, path='../data', region=''):
        # path = '../data/' maybe.
        # initialize the dataframe
        self.node_table = pd.read_csv(os.path.join(path, 'node_table_'+region+'.csv'))
        self.branch_table = pd.read_csv(os.path.join(path, 'branch_table_'+region+'.csv'))
        self.node_num = self.node_table.shape[0]-2
        self.branch_num = self.branch_table.shape[0]
        self.Ebase = np.array([2,40,40]).T
        self.gamma = 0.04 # 贴现率
        self.region = region
        if self.region == 'stu':
            self.ttl = np.array([10,10,20,20,10,20,20,20])
            self.cost = np.array([40,10,60,350,400*2,9*5,9*5,700])
            self.Pbase = np.array([0.5,0.5,2,10,1,8,8])
        elif self.region == 'off':
            self.ttl = np.array([30,20,30,30,10,20,20,20])
            self.cost = np.array([850,60,400,40,400*2,9*5,9*5,700])
            self.Pbase = np.array([2,12,3,2,1,8,8])
        elif self.region == 'tch':
            self.ttl = np.array([30,20,20,20,20,10,20,20])
            self.cost = np.array([790,380,60,60,400*2,9*5,9*5,9])
            self.Pbase = np.array([8,2,12,5,10,1,8,8])
    
    def get_in_out_branch(self, node:int):
        # 以node为终点的支路表in_branch，和以node为起点的支路表out_branch
        in_branch = self.branch_table[self.branch_table['sink']==node]
        out_branch = self.branch_table[self.branch_table['source']==node]
        return in_branch.sort_values(by='type'), out_branch.sort_values(by='type')

    def get_in_out_type(self, node:int):
        in_branch, out_branch = self.get_in_out_branch(node)
        in_type = in_branch['type'].unique()
        in_type_num = in_branch['type'].nunique()
        out_type = out_branch['type'].unique()
        out_type_num = out_branch['type'].nunique()
        return in_type_num, out_type_num, in_type, out_type

    def get_Ai(self, node:int, es: bool):
        in_branch, out_branch = self.get_in_out_branch(node)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(node)
        A = np.zeros((in_type_num+out_type_num, self.branch_num))
        # 三个储能，需要重构A
        for i in range(in_type_num):
            for j in range(in_branch.shape[0]):
                if (in_branch.iloc[j,:]['type'] == in_type[i]):
                    A[i][in_branch.iloc[j,:]['branch_no']-1] = 1

        for i in range(out_type_num):
            for j in range(out_branch.shape[0]):
                if (out_branch.iloc[j,:]['type'] == out_type[i]):
                    A[i+in_type_num][out_branch.iloc[j,:]['branch_no']-1] = -1
        
        # 如果不包括储能虚拟支路，需要做出改变，储能模块删掉最后一行和后面3列，其余模块删去后面3列
        
        if not es:
            if self.region == 'stu' or self.region == 'off':
                if node == 5 or node == 6 or node == 7:
                    A = A[:-1,:-3]
                else:
                    A = A[:,:-3]
            # tch里，储电6，储热7，储冷8
            elif self.region == 'tch':
                if node == 6 or node == 7 or node == 8:
                    A = A[:-1,:-3]
                else:
                    A = A[:,:-3]
        # 分别代表储电，储热，储冷支路，新增一行
        # stu和off里，储电5，储热6，储冷7
        # tch里，储电6，储热7，储冷8

        # print(f"A{node}, {A.shape}")
        # print(A)
        # indices_1 = np.where(A == 1)
        # print("输入端口 ", list(zip(indices_1[0], indices_1[1]+1)))
        # indices_minus_1 = np.where(A == -1)
        # print("输出端口 ", list(zip(indices_minus_1[0], indices_minus_1[1]+1)))
        return A
    
    def get_Hi(self, node:int, es:bool):
        in_branch, out_branch = self.get_in_out_branch(node)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(node)
        H = np.zeros((out_type_num, in_type_num+out_type_num))
        eta = self.node_table.iloc[2:,2:]
        eta_node = eta.iloc[node-1, 0:out_type_num].values.reshape(out_type_num,1)
        H = np.hstack((eta_node, np.eye(out_type_num)))
        if es:
            H_es = np.array([[0.92, 1/0.92, 1]])
            H_hs = H_cs = np.array([[0.95, 1/0.95, 1]])
        else:
            H_es = np.array([[0.92, 1/0.92]])
            H_hs = H_cs = np.array([[0.95, 1/0.95]])
        # 储能特殊出处理
        # stu和off里，储电5，储热6，储冷7
        if self.region == 'stu' or self.region == 'off':
            if node == 5:
                H = H_es
            elif node == 6:
                H = H_hs
            elif node == 7:
                H = H_cs
        # tch里，储电6，储热7，储冷8
        elif self.region == 'tch':
            if node == 6:
                H = H_es
            elif node == 7:
                H = H_hs
            elif node == 8:
                H = H_cs

        # print(f"H{node}, {H.shape}")
        # print(H)
        return H

    def get_Zi(self, node:int, es:bool):
        Zi = self.get_Hi(node, es) @ self.get_Ai(node, es)
        # print(f"Z{node}, {Z.shape}")
        # print(Z)
        return Zi
    
    
    def get_Z(self, es:bool):
        if not es:
            Z = None
            # 初始Z不包括储能行！！
            for node in range(1, self.node_num+1-3):
                Zi = self.get_Zi(node, es)
                # print(Zi.shape)
                if Z is None:
                    Z = Zi
                else:
                    Z = np.vstack((Z, Zi))
        else:
            if self.region == 'stu' or self.region == 'off':
                Z_es = self.get_Hi(5,es) @ self.get_Ai(5,es)
                Z_hs = self.get_Hi(6,es) @ self.get_Ai(6,es)
                Z_cs = self.get_Hi(7,es) @ self.get_Ai(7,es)
            # tch里，储电6，储热7，储冷8
            elif self.region == 'tch':
                Z_es = self.get_Hi(6,es) @ self.get_Ai(6,es)
                Z_hs = self.get_Hi(7,es) @ self.get_Ai(7,es)
                Z_cs = self.get_Hi(8,es) @ self.get_Ai(8,es)
            Z = np.vstack([Z_es[-1,:], Z_hs[-1,:], Z_cs[-1,:]])
        # self.Z = Z
        # print(f"Z, {Z.shape}")
        # print(Z)
        return Z


    def get_X(self,es:bool):
        # 系统输入branch即从-1节点(起始)出来的branch
        in_branch, out_branch = self.get_in_out_branch(-1)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(-1)

        # X = np.zeros((out_type_num, self.branch_num))
        X = np.zeros((out_type_num, self.branch_num-3))
        for i in range(out_type_num):
            for j in range(out_branch.shape[0]):
                if (out_branch.iloc[j,:]['type'] == out_type[i]):
                    X[i+in_type_num][out_branch.iloc[j,:]['branch_no']-1] = 1

        if es:
            Zs = self.get_Z(es=True)
            X = np.vstack([X, Zs[:,:-3]])

        # print(f"X, {X.shape}")
        # print(X)
        # indices_1 = np.where(X == 1)
        # print("输入端口 ", list(zip(indices_1[0], indices_1[1]+1)))
        return X
    
    def get_Y(self):
        # 系统输出branch即从0节点(汇聚)进去的branch
        in_branch, out_branch = self.get_in_out_branch(0)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(0)
        # Y = np.zeros((in_type_num, self.branch_num))
        Y = np.zeros((in_type_num, self.branch_num-3))
        for i in range(in_type_num):
            for j in range(in_branch.shape[0]):
                if (in_branch.iloc[j,:]['type'] == in_type[i]):
                    Y[i][in_branch.iloc[j,:]['branch_no']-1] = 1
        # print(f"Y", {Y.shape})
        # print(Y)
        # indices_1 = np.where(Y == 1)
        # print("输出端口 ", list(zip(indices_1[0], indices_1[1]+1)))
        self.Y = Y
        return Y
    
    def get_R(self, es):
        X, Z = self.get_X(es), self.get_Z(es=False)
        I = -np.eye(X.shape[0])
        zeros = np.zeros((Z.shape[0], X.shape[0]))
        R = np.vstack((I, zeros))
        self.R = R
        # print(R)
        return R
    
    def get_Q(self, es):
        X, Z = self.get_X(es), self.get_Z(es=False)
        self.Q = Q = np.vstack((X, Z))
        self.r = r = np.linalg.matrix_rank(Q)
        # print(f"Xshape {X.shape}")
        # print(f"Q的rank {r}")
        # print(Q)
        return Q
    
    def get_QBF(self, es):
        Q = self.get_Q(es)
        # print(Q, Q.shape)
        rref_Q, pivot_idx = rref(Q)
        self.pivot_idx = pivot_idx
        # print(rref_Q, pivot_idx)
        self.nonpivot_idx = nonpivot_idx = np.setdiff1d(np.arange(self.branch_num-3), pivot_idx).tolist()
        # print(pivot_idx)
        # print(nonpivot_idx)
        QB = Q[:, pivot_idx]
        QF = Q[:, nonpivot_idx]
        # print(QB), print(QF)
        return QB, QF
    
    def get_YBF(self):
        Y = self.get_Y()
        YB = Y[:, self.pivot_idx]
        YF = Y[:, self.nonpivot_idx]
        # print(YB), print(YF)
        return YB, YF


    def get_paramerters(self):
        X, Y, Z, R, Xs, Zs = self.get_X(es=False), self.get_Y(), self.get_Z(es=False), self.get_R(es=True), self.get_X(es=True), self.get_Z(es=True)
        QB, QF = self.get_QBF(es=True)
        # print(self.Q)
        # print(self.Q.shape)
        # print(QB.shape)
        # print(QF.shape)
        YB, YF = self.get_YBF()
        QB_inv = np.linalg.inv(QB)
        pivot_idx, nonpivot_idx = self.pivot_idx, self.nonpivot_idx
        Cin = -YB @ QB_inv @ R
        CF= YF-YB @ QB_inv @ QF

        return  X, Y, Z, R, Xs, Zs, QB, QF, YB, YF, QB_inv, pivot_idx, nonpivot_idx, Cin, CF

    def compute_C_cap(self, N1, N2=None):
        C_cap = 0
        # 等年值折算方法
        if N2 is not None:
            # print(N1.shape[0] + N2.shape[0] , len(self.cost))
            assert N1.shape[0] + N2.shape[0]  == len(self.cost)
        else:
            # print(N1.shape[0] , len(self.cost))
            assert N1.shape[0] == len(self.cost)

        for i in range(N1.shape[0]):    
            C_cap += ( self.gamma / (1-(1+self.gamma)**(self.ttl[i])) ) * self.cost[i] * N1[i]
        if N2 is not None:
            # 加上风机或光伏
            C_cap += ( self.gamma / (1-(1+self.gamma)**(self.ttl[-1])) ) * self.cost[-1] * N2

        return C_cap

if __name__ == '__main__':
    # 验证标准化建模
    std_model_stu = StdModel(path="../data", region='stu')
    print(std_model_stu.node_table)
    print(std_model_stu.node_table.shape)

    print(std_model_stu.branch_table)
    print(std_model_stu.branch_table.shape)

    ## A
    # std_model_stu.get_Ai(1), std_model_stu.get_Ai(2), std_model_stu.get_Ai(3), std_model_stu.get_Ai(4)
    # std_model_stu.get_Ai(5), std_model_stu.get_Ai(6), std_model_stu.get_Ai(7)
    # std_model_stu.get_Ai(8)

    ## X
    # std_model_stu.get_X()

    # # ## Y
    # std_model_stu.get_Y()

    # # ## H
    # # std_model_stu.get_Hi(1), std_model_stu.get_Hi(2), std_model_stu.get_Hi(3), std_model_stu.get_Hi(4)
    # # std_model_stu.get_Hi(5), std_model_stu.get_Hi(6), std_model_stu.get_Hi(7)
    # # std_model_stu.get_Hi(8)

    # # ## Z
    # std_model_stu.get_Zi(1,es=False), std_model_stu.get_Zi(2,es=False), std_model_stu.get_Zi(3,es=False), std_model_stu.get_Zi(4,es=False)
    # std_model_stu.get_Zi(5,es=False), std_model_stu.get_Zi(6,es=False), std_model_stu.get_Zi(7,es=False)
    # std_model_stu.get_Zi(1,es=True), std_model_stu.get_Zi(2,es=True), std_model_stu.get_Zi(3,es=True), std_model_stu.get_Zi(4,es=True)
    # std_model_stu.get_Zi(5,es=True), std_model_stu.get_Zi(6,es=True), std_model_stu.get_Zi(7,es=True)
    # # std_model_stu.get_Zi(8)

    std_model_stu.get_Z(es=False)
    std_model_stu.get_Z(es=True)



    ## R Q
    # std_model_stu.get_R()
    std_model_stu.get_Q(es=False)
    std_model_stu.get_Q(es=True)
    # std_model_stu.get_QBF()
    # std_model_stu.get_YBF()


