import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import Model, GRB
import cvxpy as cp


from std_modeling import StdModel


# Load data
load_e_h_c_gen_pv = pd.read_csv('table_1.CSV').values
price = pd.read_csv('table_2.CSV').values
load_e = load_e_h_c_gen_pv[:, 1]
load_h = load_e_h_c_gen_pv[:, 2]
load_c = load_e_h_c_gen_pv[:, 3]
gen_pv = load_e_h_c_gen_pv[:, 4]
price_e = price[:, 1]
price_g = price[:, 2]
print(type(price_e))

hrs = np.ones(24)
v1_max = load_e
v2_max = 50 * hrs
v3_max = 200 * hrs
v4_max = 100 * hrs
v5_max = 60 * hrs
v6_max = 60 * hrs
v7_max = 80 * hrs
v8_max = 80 * hrs
v9_max = 95 * hrs
v10_max = 95 * hrs
v11_max = 150 * hrs
v12_max = 80 * hrs


V_max = np.vstack((v1_max, v2_max, v3_max, v4_max, v5_max, v6_max, \
                   v7_max, v8_max, v9_max, v10_max, v11_max, v12_max))

# print(V_max)
std_model = StdModel(path="../data")
branch_num = std_model.branch_num
X, Y, Z, R = std_model.get_X(), std_model.get_Y(), std_model.get_Z(), std_model.get_R()
QB, QF = std_model.get_QBF()
YB, YF = std_model.get_YBF()
QB_inv = np.linalg.inv(QB)
Cin = -YB @ QB_inv @ R
CF = YF-YB @ QB_inv @ QF
pivot_idx, nonpivot_idx = std_model.pivot_idx, std_model.nonpivot_idx

V = cp.Variable((branch_num, 24))
Vin = X @ V
Vout = Y @ V
VF = V[nonpivot_idx, :]
VB = V[pivot_idx, :]

VB_min = np.zeros(VB.shape)
VB_max = V_max[pivot_idx, :]
VF_min = np.zeros(VF.shape)
VF_max = V_max[nonpivot_idx, :]

# 定义Vout_load
Vout_load = np.array([load_e, load_h, load_c])

constraints = [
    VF_min <= VF, VF <= VF_max, 
    VB_min <= VB, VB <= VB_max, 
    VB == -QB_inv @ QF @ VF - QB_inv @ R @ Vin,
    Vout_load == Cin @ Vin + CF @ VF,
    ]


obj = cp.Minimize((Vin[0, :] - gen_pv) @ price_e + Vin[1, :] @ price_g)

prob = cp.Problem(obj, constraints)

prob.solve(verbose=False, solver=cp.GUROBI)

print(f"The optimal value is {prob.value}")
# print(f"Vin = {Vin.value}")
# print(f"V = {V[8,8].value}")

print(f"电输出 = {V[0,:].value + V[5,:].value}")
print(f"电负荷 = {Vout_load[0,:]}")
print(f"热输出 = {V[7,:].value + V[9,:].value}")
print(f"热负荷 = {Vout_load[1,:]}")
print(f"冷输出 = {V[10,:].value + V[11,:].value}")
print(f"冷负荷 = {Vout_load[2,:]}")
print(f"所有输出 = {Cin @ Vin.value + CF @ VF.value}")

hours = np.arange(24)
plt.figure(figsize=(12, 6))
plt.plot(hours, Vin[0, :].value, label='e')
plt.plot(hours, Vin[1, :].value, label='g')
plt.xlabel('Hour (h)')
plt.ylabel('Demand (MW)')
plt.title('Year-round Load')
plt.legend()
plt.show()

