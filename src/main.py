import os
import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import Model, GRB
import cvxpy as cp

from data import EnergyData
from std_modeling import StdModel

"""
将清华大学分区建模为三个能量枢纽， 对各能量枢纽内的设备进行选型，规划，
分析清华大学在未来碳中和下能源利用方式
"""
# load data
data_loader = EnergyData(path="../data")
data_loader = data_loader.get_2d_array().get_8760()

heating_load = data_loader.load_heating_xs_8760 + data_loader.load_heating_jg_8760 + data_loader.load_heating_jxbg_8760
cooling_load = data_loader.load_cooling_xs_8760 + data_loader.load_cooling_jg_8760 + data_loader.load_cooling_jxbg_8760
electricity_load = data_loader.load_electricity_xs_8760 + data_loader.load_electricity_jg_8760 + data_loader.load_electricity_jxbg_8760

# 1-3行分别为电-热-冷8760小时负荷需求
V_load_stu = np.vstack((data_loader.load_electricity_xs_8760, data_loader.load_heating_xs_8760, data_loader.load_cooling_xs_8760))
V_load_off = np.vstack((data_loader.load_electricity_jxbg_8760, data_loader.load_heating_jxbg_8760, data_loader.load_cooling_jxbg_8760))
V_load_tch = np.vstack((data_loader.load_electricity_jg_8760, data_loader.load_heating_jg_8760, data_loader.load_cooling_jg_8760))

# load standard model
std_model_stu = StdModel(path='../data', region='stu')
std_model_off = StdModel(path='../data', region='off')
std_model_tch = StdModel(path='../data', region='tch')

X_stu, Y_stu, Z_stu, R_stu = std_model_stu.get_X(), std_model_stu.get_Y(), std_model_stu.get_Z(), std_model_stu.get_R()
QB_stu, QF_stu = std_model_stu.get_QBF()
YB_stu, YF_stu = std_model_stu.get_YBF()
QB_stu_inv = np.linalg.inv(QB_stu)
pivot_idx_stu, nonpivot_idx_stu = std_model_stu.pivot_idx, std_model_stu.nonpivot_idx
Cin_stu = -YB_stu @ QB_stu_inv @ R_stu
CF_stu = YF_stu-YB_stu @ QB_stu_inv @ QF_stu

X_off, Y_off, Z_off, R_off = std_model_off.get_X(), std_model_off.get_Y(), std_model_off.get_Z(), std_model_off.get_R()
QB_off, QF_off = std_model_off.get_QBF()
YB_off, YF_off = std_model_off.get_YBF()
QB_off_inv = np.linalg.inv(QB_off)
pivot_idx_off, nonpivot_idx_off = std_model_off.pivot_idx, std_model_off.nonpivot_idx
Cin_off = -YB_off @ QB_off_inv @ R_off
CF_off = YF_off-YB_off @ QB_off_inv @ QF_off

X_tch, Y_tch, Z_tch, R_tch = std_model_tch.get_X(), std_model_tch.get_Y(), std_model_tch.get_Z(), std_model_tch.get_R()
QB_tch, QF_tch = std_model_tch.get_QBF()
YB_tch, YF_tch = std_model_tch.get_YBF()
QB_tch_inv = np.linalg.inv(QB_tch)
pivot_idx_tch, nonpivot_idx_tch = std_model_tch.pivot_idx, std_model_tch.nonpivot_idx
Cin_tch = -YB_tch @ QB_tch_inv @ R_tch
CF_tch = YF_tch-YB_tch @ QB_tch_inv @ QF_tch


### build a opt model
hrs = np.ones(8760)
# 8760小时，分区中，每个支路的功率流变量
V_stu = cp.Variable((std_model_stu.branch_num, 8760), name='V in stu')
V_off = cp.Variable((std_model_off.branch_num, 8760), name='V in off')
V_tch = cp.Variable((std_model_tch.branch_num, 8760), name='V in tch')
# 分区中，每个设备的个数
N_stu = cp.Variable(std_model_stu.node_num, integer=True, name='device number in stu')
N_off = cp.Variable(std_model_off.node_num, integer=True, name='device number in off')
N_tch = cp.Variable(std_model_tch.node_num, integer=True, name='device number in tch')
N_pv = cp.Variable(1, integer=True, name='pv number')
N_wt = cp.Variable(1, integer=True, name='wt number')

assert N_stu.shape == std_model_stu.Pbase.shape and N_off.shape == std_model_off.Pbase.shape \
        and N_tch.shape == std_model_tch.Pbase.shape
# element-wise multiply, 每个设备的容量
device_stu_max = cp.multiply(N_stu, std_model_stu.Pbase)
device_off_max = cp.multiply(N_off, std_model_off.Pbase)
device_tch_max = cp.multiply(N_tch, std_model_tch.Pbase)

V_stu_max = cp.Variable(V_stu.shape)
V_off_max = cp.Variable(V_off.shape)
V_tch_max = cp.Variable(V_tch.shape)

#%%
constraints = []
## 每条支路加入最大功率限制，容量为输入能量的最大值
## 学生区，支路功率约束
constraints += [
#1 压缩式制冷机A输入功率限制
V_stu_max[0,:] == cp.multiply(device_stu_max[0], hrs),
V_stu_max[6,:] == cp.multiply(device_stu_max[0], hrs),
#2 电制气输入功率限制
V_stu_max[1,:] == cp.multiply(device_stu_max[1], hrs),
V_stu_max[7,:] == cp.multiply(device_stu_max[1], hrs),
#3 电锅炉输入功率限制
V_stu_max[2,:] == cp.multiply(device_stu_max[2], hrs),
V_stu_max[8,:] == cp.multiply(device_stu_max[2], hrs),
#4 热泵输入功率限制
V_stu_max[3,:] == cp.multiply(device_stu_max[3], hrs),
V_stu_max[9,:] == cp.multiply(device_stu_max[3], hrs),
#5 储电输入限制
V_stu_max[5,:] == cp.multiply(device_stu_max[4], hrs),
#5 储电输出限制 678910
V_stu_max[6,:] == cp.minimum(cp.multiply(device_stu_max[0], hrs), cp.multiply(device_stu_max[4], hrs)),
V_stu_max[7,:] == cp.minimum(cp.multiply(device_stu_max[1], hrs), cp.multiply(device_stu_max[4], hrs)),
V_stu_max[8,:] == cp.minimum(cp.multiply(device_stu_max[2], hrs), cp.multiply(device_stu_max[4], hrs)),
V_stu_max[9,:] == cp.minimum(cp.multiply(device_stu_max[3], hrs), cp.multiply(device_stu_max[4], hrs)),
V_stu_max[10,:] == cp.multiply(device_stu_max[4], hrs),
#6 储热输入限制
V_stu_max[13,:] == cp.multiply(device_stu_max[5], hrs),
V_stu_max[16,:] == cp.multiply(device_stu_max[5], hrs),
V_stu_max[18,:] == cp.multiply(device_stu_max[5], hrs),
#6 储热输出限制
V_stu_max[14,:] ==  cp.multiply(device_stu_max[5], hrs),
#7 储冷输入限制
V_stu_max[20,:] == cp.multiply(device_stu_max[6], hrs),
#7 储冷输出限制
V_stu_max[21,:] == cp.multiply(device_stu_max[6], hrs),
]

## 教学办公区，支路功率约束
constraints += [
#1 热电联产机组A输入功率限制
V_off_max[7,:] == cp.multiply(device_off_max[0], hrs),
#2 吸收式制冷机组输入功率限制
V_off_max[10,:] == cp.multiply(device_off_max[1], hrs),
V_off_max[12,:] == cp.multiply(device_off_max[1], hrs),
V_off_max[16,:] == cp.multiply(device_off_max[1], hrs),
V_off_max[19,:] == cp.multiply(device_off_max[1], hrs),
#3 冷热电联供输入功率限制
V_off_max[8,:] == cp.multiply(device_off_max[2], hrs),
#4 燃气蒸汽锅炉输入功率限制
V_off_max[9,:] == cp.multiply(device_off_max[3], hrs),
#5 储电输入功率限制
V_off_max[1,:] == cp.multiply(device_off_max[4], hrs),
V_off_max[3,:] == cp.multiply(device_off_max[4], hrs),
V_off_max[5,:] == cp.multiply(device_off_max[4], hrs),
#5 储电输出功率限制
V_off_max[6,:] == cp.multiply(device_off_max[4], hrs),
#6 储热输入功率限制
V_off_max[11,:] == cp.multiply(device_off_max[5], hrs),
V_off_max[17,:] == cp.multiply(device_off_max[5], hrs),
V_off_max[20,:] == cp.multiply(device_off_max[5], hrs),
#6 储热输出功率限制
V_off_max[12,:] == cp.multiply(device_off_max[5], hrs),
V_off_max[14,:] == cp.multiply(device_off_max[5], hrs),
#7 蓄冷输入功率限制
V_off_max[22,:] == cp.multiply(device_off_max[6], hrs),
V_off_max[25,:] == cp.multiply(device_off_max[6], hrs),
#7 蓄冷输出功率限制
V_off_max[23,:] == cp.multiply(device_off_max[6], hrs),
]

## 教学区，支路功率约束
constraints += [
#1 热电联产机组B输入功率限制
V_tch_max[19,:] == cp.multiply(device_tch_max[0], hrs),
#2 地源热泵A输入功率限制
V_tch_max[1,:] == cp.multiply(device_tch_max[1], hrs),
V_tch_max[5,:] == cp.multiply(device_tch_max[1], hrs),
V_tch_max[9,:] == cp.multiply(device_tch_max[1], hrs),
V_tch_max[13,:] == cp.multiply(device_tch_max[1], hrs),
V_tch_max[17,:] == cp.multiply(device_tch_max[1], hrs),
#3 压缩式制冷机组B输入功率限制
V_tch_max[2,:] == cp.multiply(device_tch_max[2], hrs),
V_tch_max[6,:] == cp.multiply(device_tch_max[2], hrs),
V_tch_max[10,:] == cp.multiply(device_tch_max[2], hrs),
V_tch_max[14,:] == cp.multiply(device_tch_max[2], hrs),
V_tch_max[18,:] == cp.multiply(device_tch_max[2], hrs),
#4 燃气轮机输入功率限制
V_tch_max[20,:] == cp.multiply(device_tch_max[3], hrs),
#5 内燃机输入功率限制
V_tch_max[21,:] == cp.multiply(device_tch_max[4], hrs),
#6 储电输入功率限制
V_tch_max[3,:] == cp.multiply(device_tch_max[5], hrs),
V_tch_max[7,:] == cp.multiply(device_tch_max[5], hrs),
V_tch_max[11,:] == cp.multiply(device_tch_max[5], hrs),
V_tch_max[15,:] == cp.multiply(device_tch_max[5], hrs),
#6 储电输出功率限制
V_tch_max[16,:] == cp.multiply(device_tch_max[5], hrs),
V_tch_max[17,:] == cp.minimum(V_tch_max[17,:], cp.multiply(device_tch_max[5], hrs)),
V_tch_max[18,:] == cp.minimum(V_tch_max[18,:], cp.multiply(device_tch_max[5], hrs)),
#7 储热输入功率限制
V_tch_max[23,:] == cp.multiply(device_tch_max[6], hrs),
V_tch_max[26,:] == cp.multiply(device_tch_max[6], hrs),
V_tch_max[28,:] == cp.multiply(device_tch_max[6], hrs),
V_tch_max[30,:] == cp.multiply(device_tch_max[6], hrs),
#7 储热输出功率限制
V_tch_max[24,:] == cp.multiply(device_tch_max[6], hrs),
#8 蓄冷输入功率限制
V_tch_max[32,:] == cp.multiply(device_tch_max[7], hrs),
#8 蓄冷输出功率限制
V_tch_max[33,:] == cp.multiply(device_tch_max[7], hrs),
]

V_stu_in, V_stu_out = X_stu @ V_stu, Y_stu @ V_stu
V_off_in, V_off_out = X_off @ V_off, Y_off @ V_off
V_tch_in, V_tch_out = X_tch @ V_tch, Y_tch @ V_tch

VB_stu, VF_stu = V_stu[pivot_idx_stu, :], V_stu[nonpivot_idx_stu, :]
VB_off, VF_off = V_off[pivot_idx_off, :], V_off[nonpivot_idx_off, :]
VB_tch, VF_tch = V_tch[pivot_idx_tch, :], V_tch[nonpivot_idx_tch, :]

VB_stu_min, VB_stu_max = np.zeros(VB_stu.shape), V_stu_max[pivot_idx_stu]
VF_stu_min, VF_stu_max = np.zeros(VF_stu.shape), V_stu_max[nonpivot_idx_stu]
VB_off_min, VB_off_max = np.zeros(VB_off.shape), V_off_max[pivot_idx_off]
VF_off_min, VF_off_max = np.zeros(VF_off.shape), V_off_max[nonpivot_idx_off]
VB_tch_min, VB_tch_max = np.zeros(VB_tch.shape), V_tch_max[pivot_idx_tch]
VF_tch_min, VF_tch_max = np.zeros(VF_tch.shape), V_tch_max[nonpivot_idx_tch]

# 分区能源枢纽约束
constraints += [
    VF_stu_min <= VF_stu, VF_off_min <= VF_off, VF_tch_min <= VF_tch,
    VF_stu <= VF_stu_max, VF_off <= VF_off_max, VF_tch <= VF_tch_max,
    VB_stu_min <= VB_stu, VB_off_min <= VB_off, VB_tch_min <= VB_tch,
    VB_stu <= VB_stu_max, VB_off <= VB_off_max, VB_tch <= VB_tch_max,
    VB_stu == -QB_stu_inv @ QF_stu @ VF_stu - QB_stu_inv @ R_stu @ V_stu_in,
    VB_off == -QB_off_inv @ QF_off @ VF_off - QB_off_inv @ R_off @ V_off_in,
    VB_tch == -QB_tch_inv @ QF_tch @ VF_tch - QB_tch_inv @ R_tch @ V_tch_in,
    # V_load_stu == Cin_stu @ V_stu_in + CF_stu @ VF_stu,
    ]
V_gen_stu = Cin_stu @ V_stu_in + CF_stu @ VF_stu # (4, 8760) 气输出，而非负荷
V_gen_off = Cin_off @ V_off_in + CF_off @ VF_off # (3, 8760) 电热冷，没有气输出
V_gen_tch = Cin_tch @ V_tch_in + CF_tch @ VF_tch # (3, 8760)

V_gen_for_load_stu = cp.Variable((3, 8760), name='Vout for load in stu')
V_gen_for_out_stu = cp.Variable((3, 8760), name='Vout for out in stu')
V_gen_for_load_off = cp.Variable((3, 8760), name='Vout for load in off')
V_gen_for_out_off = cp.Variable((3, 8760), name='Vout for out in off')
V_gen_for_load_tch = cp.Variable((3, 8760), name='Vout for load in tch')
V_gen_for_out_tch = cp.Variable((3, 8760), name='Vout for out in tch')

constraints += [
    # 电气热
    # 学生区输出拆分，没有热输出
    V_gen_stu[[0,2,3],:] == V_gen_for_load_stu + V_gen_for_out_stu,
    V_gen_for_out_stu[2,:] == 0,
    # 教学办公区输出拆分，没有气输出
    V_gen_off == V_gen_for_load_off + V_gen_for_out_off,
    V_gen_for_out_off[1,:] == 0,
    # 教工区没有输出
    V_gen_tch == V_gen_for_load_tch + V_gen_for_out_tch,
    V_gen_for_out_tch == 0,
    V_gen_for_load_stu <= V_load_stu,
    V_gen_for_load_off <= V_load_off,
    V_gen_for_load_tch <= V_load_tch,
    ]



## 储能约束
eta_e, eta_h, eta_c = 0.92, 0.95, 0.95
SOC_stu = cp.Variable((3, 8760+1), name='SOC in stu')
SOC_off = cp.Variable((3, 8760+1), name='SOC in off')
SOC_tch = cp.Variable((3, 8760+1), name='SOC in tch')
ES_stu_in, ES_stu_out = cp.Variable((1, 8760), name='input power of ES in stu'), cp.Variable((1, 8760), name='output power of ES in stu')
ES_off_in, ES_off_out = cp.Variable((1, 8760), name='input power of ES in off'), cp.Variable((1, 8760), name='output power of ES in off')
ES_tch_in, ES_tch_out = cp.Variable((1, 8760), name='input power of ES in tch'), cp.Variable((1, 8760), name='output power of ES in tch')
HS_stu_in, HS_stu_out = cp.Variable((1, 8760), name='input power of HS in stu'), cp.Variable((1, 8760), name='output power of HS in stu')
HS_off_in, HS_off_out = cp.Variable((1, 8760), name='input power of HS in off'), cp.Variable((1, 8760), name='output power of HS in off')
HS_tch_in, HS_tch_out = cp.Variable((1, 8760), name='input power of HS in tch'), cp.Variable((1, 8760), name='output power of HS in tch')
CS_stu_in, CS_stu_out = cp.Variable((1, 8760), name='input power of CS in stu'), cp.Variable((1, 8760), name='output power of CS in stu')
CS_off_in, CS_off_out = cp.Variable((1, 8760), name='input power of CS in off'), cp.Variable((1, 8760), name='output power of CS in off')
CS_tch_in, CS_tch_out = cp.Variable((1, 8760), name='input power of CS in tch'), cp.Variable((1, 8760), name='output power of CS in tch')
constraints += [
    SOC_stu[:,0] == 0.5, SOC_off[:,0] == 0.5, SOC_tch[:,0] == 0.5,
]
for day in range(365):
    for hour in range(24):
        cur_hour = 24*day + hour
        constraints += [
            # SOC 上下限
            0 <= SOC_stu, SOC_stu <= 1, 0 <= SOC_off, SOC_off <= 1, 0 <= SOC_tch, SOC_tch <= 1,
            # 储能日内平衡
            SOC_stu[:,24*day] == 0.5, SOC_off[:,24*day] == 0.5, SOC_tch[:,24*day] == 0.5,
            # 储能递推约束
            SOC_stu[0,cur_hour+1] == SOC_stu[0,cur_hour] + eta_e * ES_stu_in - 1/eta_e * ES_stu_out,
            SOC_stu[1,cur_hour+1] == SOC_stu[1,cur_hour] + eta_h * HS_stu_in - 1/eta_h * HS_stu_out,
            SOC_stu[2,cur_hour+1] == SOC_stu[2,cur_hour] + eta_c * CS_stu_in - 1/eta_c * CS_stu_out,
            SOC_off[0,cur_hour+1] == SOC_off[0,cur_hour] + eta_e * ES_off_in - 1/eta_e * ES_off_out,
            SOC_off[1,cur_hour+1] == SOC_off[1,cur_hour] + eta_h * HS_off_in - 1/eta_h * HS_off_out,
            SOC_off[2,cur_hour+1] == SOC_off[2,cur_hour] + eta_c * CS_off_in - 1/eta_c * CS_off_out,
            SOC_tch[0,cur_hour+1] == SOC_tch[0,cur_hour] + eta_e * ES_tch_in - 1/eta_e * ES_tch_out,
            SOC_tch[1,cur_hour+1] == SOC_tch[1,cur_hour] + eta_h * HS_tch_in - 1/eta_h * HS_tch_out,
            SOC_tch[2,cur_hour+1] == SOC_tch[2,cur_hour] + eta_c * CS_tch_in - 1/eta_c * CS_tch_out,
        ]

ele_stu_purchased = cp.Variable(8760, name='purchased power in stu')
ele_off_purchased = cp.Variable(8760, name='purchased power in stu')
ele_tch_purchased = cp.Variable(8760, name='purchased power in stu')
ele_stu2off = cp.Variable(8760, name='power stu to off')
ele_stu2tch = cp.Variable(8760, name='power stu to tch')
ele_off2tch = cp.Variable(8760, name='power off to tch')
gas_purchased = cp.Variable(8760, name='purchased gas')

print(f"V_gen_for_out_off{V_gen_for_out_off.shape}")
print(f"V_stu_in{V_stu_in.shape}")
print(f"V_tch_in{V_tch_in.shape}")
# 母线平衡
constraints += [
    # 电母线平衡
    ele_stu_purchased + V_gen_for_out_stu[0,:] == V_stu_in[0,:] + ele_stu2off + ele_stu2tch,
    ele_off_purchased + ele_stu2off + V_gen_for_out_off[0,:] == V_off_in[0,:] + ele_off2tch,
    ele_tch_purchased + ele_stu2tch + ele_off2tch == V_tch_in[0,:],
    # 气母线平衡
    gas_purchased + V_gen_for_out_stu[1,:] == V_off_in[1,:] + V_tch_in[1,:],
    # 热母线平衡
    V_gen_for_out_off[2,:] == V_stu_in[1,:] + V_tch_in[2,:] 
]

#%%

# 电-热-冷 失负荷矩阵
unmet_stu = cp.sum(V_load_stu - V_gen_for_load_stu)
unmet_off = cp.sum(V_load_off - V_gen_for_load_off)
unmet_tch = cp.sum(V_load_tch - V_gen_for_load_tch)
unmet = unmet_stu + unmet_off + unmet_tch

# V_stu_in (2, 8760) 电热输入
# V_off_in (2, 8760) 电气输入
# V_tch_in (3, 8760) 电气热输入
pv_stu_in = N_pv * data_loader.supply_pv_8760
ele_purchased_stu = V_stu_in[0,:] - pv_stu_in
gas_purchased_stu = 0

wt_off_in = N_wt * data_loader.supply_wt_8760
ele_purchased_off = V_off_in[0,:] - pv_stu_in
gas_purchased_off = V_off_in[1,:]

ele_purchased_tch = V_tch_in[0,:]
gas_purchased_tch = V_tch_in[1,:]

ele_purchased = ele_purchased_stu + ele_purchased_off + ele_purchased_tch
gas_purchased = gas_purchased_stu + gas_purchased_off + gas_purchased_tch
C_ele = data_loader.price_electricity_8760 @ ele_purchased
C_gas = data_loader.price_gas_8760 @ gas_purchased
C_unmet = 50 * unmet
C_op = C_ele + C_gas + C_unmet

C_cap_stu = N_stu @ std_model_stu.cost[:-1] + N_pv * std_model_stu.cost[-1]
C_cap_off = N_off @ std_model_off.cost[:-1] + N_wt * std_model_off.cost[-1]
C_cap_tch = N_tch @ std_model_tch.cost
C_cap = C_cap_stu + C_cap_off + C_cap_tch

carbon = ele_purchased @ data_loader.carbon_factor_8760 + gas_purchased * 0.0021650 # tCO2
carbon_exceed = cp.Variable()

constraints += [
    carbon_exceed >= cp.sum(carbon) - 3e5,
    carbon_exceed >= 0,
]
C_carbon = 200*carbon_exceed
    
C_total = C_cap + C_op + C_carbon



if __name__ == '__main__':
    obj = cp.Minimize(C_total)
    problem = cp.Problem(obj, constraints)
    problem.solve(verbose=False, solver=cp.GUROBI)

    print(f"The optimal value is {problem.value}")



