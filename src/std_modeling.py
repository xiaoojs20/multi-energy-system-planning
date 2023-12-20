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
        self.Ebase = np.array([2,40,40])
        if region == 'stu':
            self.cost = np.array([40,10,60,350,400*2,9*5,9*5,700])
            self.Pbase = np.array([0.5,0.5,2,10,1,8,8])
        elif region == 'off':
            self.cost = np.array([850,60,400,40,400*2,9*5,9*5,700])
            self.Pbase = np.array([2,12,3,2,1,8,8])
        elif region == 'tch':
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

    def get_Ai(self, node:int):
        in_branch, out_branch = self.get_in_out_branch(node)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(node)
        A = np.zeros((in_type_num+out_type_num, self.branch_num))
        
        for i in range(in_type_num):
            for j in range(in_branch.shape[0]):
                if (in_branch.iloc[j,:]['type'] == in_type[i]):
                    A[i][in_branch.iloc[j,:]['branch_no']-1] = 1

        for i in range(out_type_num):
            for j in range(out_branch.shape[0]):
                if (out_branch.iloc[j,:]['type'] == out_type[i]):
                    A[i+in_type_num][out_branch.iloc[j,:]['branch_no']-1] = -1
        # print(f"A{node}")
        # print(A)
        # indices_1 = np.where(A == 1)
        # print("输入端口 ", list(zip(indices_1[0], indices_1[1]+1)))
        # indices_minus_1 = np.where(A == -1)
        # print("输出端口 ", list(zip(indices_minus_1[0], indices_minus_1[1]+1)))
        return A
    
    def get_Hi(self, node:int):
        in_branch, out_branch = self.get_in_out_branch(node)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(node)
        H = np.zeros((out_type_num, in_type_num+out_type_num))
        eta = self.node_table.iloc[2:,2:]
        eta_node = eta.iloc[node-1, 0:out_type_num].values.reshape(out_type_num,1)
        H = np.hstack((eta_node, np.eye(out_type_num)))
        # print(f"H{node}")
        # print(H)
        return H

    def get_Zi(self, node:int):
        Z = self.get_Hi(node) @ self.get_Ai(node)
        # print(f"Z{node}")
        # print(Z)
        return Z
    
    def get_Z(self):
        Z = None
        for node in range(1, self.node_num+1):
            Zi = self.get_Zi(node)
            if Z is None:
                Z = Zi
            else:
                Z = np.vstack((Z, Zi))
        self.Z = Z
        # print(Z)
        return Z
    
    def get_X(self):
        # 系统输入branch即从-1节点(起始)出来的branch
        in_branch, out_branch = self.get_in_out_branch(-1)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(-1)
        X = np.zeros((out_type_num, self.branch_num))
        for i in range(out_type_num):
            for j in range(out_branch.shape[0]):
                if (out_branch.iloc[j,:]['type'] == out_type[i]):
                    X[i+in_type_num][out_branch.iloc[j,:]['branch_no']-1] = 1
        # print(f"X")
        # print(X)
        # indices_1 = np.where(X == 1)
        # print("输入端口 ", list(zip(indices_1[0], indices_1[1]+1)))
        self.X = X
        return X
    
    def get_Y(self):
        # 系统输出branch即从0节点(汇聚)进去的branch
        in_branch, out_branch = self.get_in_out_branch(0)
        in_type_num, out_type_num, in_type, out_type = self.get_in_out_type(0)
        Y = np.zeros((in_type_num, self.branch_num))
        for i in range(in_type_num):
            for j in range(in_branch.shape[0]):
                if (in_branch.iloc[j,:]['type'] == in_type[i]):
                    Y[i][in_branch.iloc[j,:]['branch_no']-1] = 1
        # print(f"Y")
        # print(Y)
        # indices_1 = np.where(Y == 1)
        # print("输出端口 ", list(zip(indices_1[0], indices_1[1]+1)))
        self.Y = Y
        return Y
    
    def get_R(self):
        X, Z = self.get_X(), self.get_Z()
        I = -np.eye(X.shape[0])
        zeros = np.zeros((Z.shape[0], X.shape[0]))
        R = np.vstack((I, zeros))
        self.R = R
        # print(R)
        return R
    
    def get_Q(self):
        X, Z = self.get_X(), self.get_Z()
        self.Q = Q = np.vstack((X, Z))
        self.r = r = np.linalg.matrix_rank(Q)
        # print(Q)
        return Q
    
    def get_QBF(self):
        X, Z = self.get_X(), self.get_Z()
        self.Q = Q = np.vstack((X, Z))
        rref_Q, pivot_idx = rref(Q)
        self.pivot_idx = pivot_idx
        # print(rref_Q, pivot_idx)
        self.nonpivot_idx = nonpivot_idx = np.setdiff1d(np.arange(self.branch_num), pivot_idx).tolist()
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


if __name__ == '__main__':
    # 验证标准化建模
    std_model_stu = StdModel(path="../data", region='stu')
    print(std_model_stu.node_table)
    print(std_model_stu.node_table.shape)

    print(std_model_stu.branch_table)
    print(std_model_stu.branch_table.shape)

    ## A
    std_model_stu.get_Ai(1), std_model_stu.get_Ai(2), std_model_stu.get_Ai(3), std_model_stu.get_Ai(4)
    std_model_stu.get_Ai(5), std_model_stu.get_Ai(6), std_model_stu.get_Ai(7)
    # std_model_stu.get_Ai(8)

    ## X
    std_model_stu.get_X()

    ## Y
    std_model_stu.get_Y()

    ## H
    std_model_stu.get_Hi(1), std_model_stu.get_Hi(2), std_model_stu.get_Hi(3), std_model_stu.get_Hi(4)
    std_model_stu.get_Hi(5), std_model_stu.get_Hi(6), std_model_stu.get_Hi(7)
    # std_model_stu.get_Hi(8)

    ## Z
    std_model_stu.get_Zi(1), std_model_stu.get_Zi(2), std_model_stu.get_Zi(3), std_model_stu.get_Zi(4)
    std_model_stu.get_Zi(5), std_model_stu.get_Zi(6), std_model_stu.get_Zi(7)
    # std_model_stu.get_Zi(8)

    # std_model_stu.get_Z()

    ## R Q
    # std_model_stu.get_R()
    # std_model_stu.get_Q()
    # std_model_stu.get_QBF()
    # std_model_stu.get_YBF()


