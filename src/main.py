import os
import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import Model, GRB
import cvxpy as cp

from utils import save_N, load_N, save_V, load_V
from data import EnergyData
from std_modeling import StdModel

"""
将清华大学分区建模为三个能量枢纽， 对各能量枢纽内的设备进行选型，规划，
分析清华大学在未来碳中和下能源利用方式
"""
# =============================================================================================
# ==================================  data preparing ==========================================
# =============================================================================================
# init data
CARBON_MAX = 10e4
CARBON_PER_GAS = 0.0021650  # tCO_2/m^3
GAS2MWH = 100 # 1 m^3 = 0.01 MWH
UNMET_PENALTY = 50e4 # 失负荷惩罚 50万元/MWh
CARBON_PENALTY = 200 # CO2排放惩罚 200/tCO2


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
std_model_stu, std_model_off, std_model_tch = StdModel(path='../data', region='stu'), StdModel(path='../data', region='off'), StdModel(path='../data', region='tch')

X_stu, Y_stu, Z_stu, R_stu, Xs_stu, Zs_stu, QB_stu, QF_stu, YB_stu, YF_stu, QB_stu_inv, pivot_idx_stu, nonpivot_idx_stu, Cin_stu, CF_stu = std_model_stu.get_paramerters()
X_off, Y_off, Z_off, R_off, Xs_off, Zs_off, QB_off, QF_off, YB_off, YF_off, QB_off_inv, pivot_idx_off, nonpivot_idx_off, Cin_off, CF_off = std_model_off.get_paramerters()
X_tch, Y_tch, Z_tch, R_tch, Xs_tch, Zs_tch, QB_tch, QF_tch, YB_tch, YF_tch, QB_tch_inv, pivot_idx_tch, nonpivot_idx_tch, Cin_tch, CF_tch = std_model_tch.get_paramerters()

# =============================================================================================
# ==================================  build a opt model =======================================
# =============================================================================================

# =============================================================================================
# =================================== Create Variable =========================================
# =============================================================================================
hrs = np.ones(8760)
hrs_ = np.ones((1,8761))
# 8760小时，分区中，每个支路的功率流变量，带3个储能虚拟支路
V_stu = cp.Variable((std_model_stu.branch_num, 8760), name='V in stu') # (25, 8760)
V_off = cp.Variable((std_model_off.branch_num, 8760), name='V in off') # (29, 8760)
V_tch = cp.Variable((std_model_tch.branch_num, 8760), name='V in tch') # (37, 8760)
V_stu_no_storage, V_stu_storage  = V_stu[:-3, :], V_stu[-3:, :] # 22 + 3
V_off_no_storage, V_off_storage  = V_off[:-3, :], V_off[-3:, :] # 26 + 3
V_tch_no_storage, V_tch_storage  = V_tch[:-3, :], V_tch[-3:, :] # 34 + 3

# 分区中，每个设备的个数
N_stu = cp.Variable(std_model_stu.node_num, integer=True, name='device number in stu')
N_off = cp.Variable(std_model_off.node_num, integer=True, name='device number in off')
N_tch = cp.Variable(std_model_tch.node_num, integer=True, name='device number in tch')
N_pv = cp.Variable(1, integer=True, name='pv number')
N_wt = cp.Variable(1, integer=True, name='wt number')

assert N_stu.shape == std_model_stu.Pbase[:-1].shape and N_off.shape == std_model_off.Pbase[:-1].shape \
        and N_tch.shape == std_model_tch.Pbase.shape
# 学生区和教学办公区，暂时不包括光伏和风机
# element-wise multiply, 每个设备的功率容量
device_stu_max = cp.multiply(N_stu, std_model_stu.Pbase[:-1])
device_off_max = cp.multiply(N_off, std_model_off.Pbase[:-1])
device_tch_max = cp.multiply(N_tch, std_model_tch.Pbase)

# element-wise multiply, 每个储能的能量容量 (3,)
es_stu_max = cp.multiply(N_stu[-3:], std_model_stu.Ebase)
es_off_max = cp.multiply(N_off[-3:], std_model_off.Ebase)
es_tch_max = cp.multiply(N_tch[-3:], std_model_tch.Ebase)

V_stu_max = cp.Variable(V_stu.shape, name='V stu max') # (25, 8760)
V_off_max = cp.Variable(V_off.shape, name='V off max') # (29, 8760)
V_tch_max = cp.Variable(V_tch.shape, name='V tch max') # (37, 8760)

# (2, 8760)电热输入 (4, 8760)
V_stu_in_no_storage, V_stu_out = X_stu @ V_stu_no_storage, Y_stu @ V_stu_no_storage
# (2, 8760)电气输入 (3, 8760)
V_off_in_no_storage, V_off_out = X_off @ V_off_no_storage, Y_off @ V_off_no_storage
# (3, 8760)电气热输入 (3, 8760)
V_tch_in_no_storage, V_tch_out = X_tch @ V_tch_no_storage, Y_tch @ V_tch_no_storage

## 增加储能后！！！
# (5, 8760)
V_stu_in = cp.vstack([V_stu_in_no_storage, V_stu[-3:,:]])
V_off_in = cp.vstack([V_off_in_no_storage, V_off[-3:,:]])
V_tch_in = cp.vstack([V_tch_in_no_storage, V_tch[-3:,:]])

# print(f"V_stu_in_storage {V_stu_in_storage.shape}")
VB_stu, VF_stu = V_stu_no_storage[pivot_idx_stu, :], V_stu_no_storage[nonpivot_idx_stu, :]
VB_off, VF_off = V_off_no_storage[pivot_idx_off, :], V_off_no_storage[nonpivot_idx_off, :]
VB_tch, VF_tch = V_tch_no_storage[pivot_idx_tch, :], V_tch_no_storage[nonpivot_idx_tch, :]

VB_stu_min, VB_stu_max = np.zeros(VB_stu.shape), V_stu_max[pivot_idx_stu]
VF_stu_min, VF_stu_max = np.zeros(VF_stu.shape), V_stu_max[nonpivot_idx_stu]
VB_off_min, VB_off_max = np.zeros(VB_off.shape), V_off_max[pivot_idx_off]
VF_off_min, VF_off_max = np.zeros(VF_off.shape), V_off_max[nonpivot_idx_off]
VB_tch_min, VB_tch_max = np.zeros(VB_tch.shape), V_tch_max[pivot_idx_tch]
VF_tch_min, VF_tch_max = np.zeros(VF_tch.shape), V_tch_max[nonpivot_idx_tch]

# Cin (4, 5)
# CF (4, 13)
V_gen_stu = Cin_stu @ V_stu_in + CF_stu @ VF_stu # (4, 8760) 气输出，而非负荷
V_gen_off = Cin_off @ V_off_in + CF_off @ VF_off # (3, 8760) 电热冷，没有气产出
V_gen_tch = Cin_tch @ V_tch_in + CF_tch @ VF_tch # (3, 8760) 电热冷，没有气产出

V_gen_for_load_stu = cp.Variable((4, 8760), name='Vout for load in stu')
V_gen_for_out_stu = cp.Variable((4, 8760), name='Vout for out in stu')
V_gen_for_load_off = cp.Variable((3, 8760), name='Vout for load in off')
V_gen_for_out_off = cp.Variable((3, 8760), name='Vout for out in off')
V_gen_for_load_tch = cp.Variable((3, 8760), name='Vout for load in tch')
V_gen_for_out_tch = cp.Variable((3, 8760), name='Vout for out in tch')

es_stu = cp.Variable((3, 8760+1), name='es in stu') # 各行：电热冷储能能量
es_off = cp.Variable((3, 8760+1), name='es in off')
es_tch = cp.Variable((3, 8760+1), name='es in tch')

ele_stu_purchased = cp.Variable(8760, name='purchased power in stu')
ele_off_purchased = cp.Variable(8760, name='purchased power in off')
ele_tch_purchased = cp.Variable(8760, name='purchased power in tch')
ele_purchased = cp.Variable(8760, name='purchased power total')
ele_stu2off = cp.Variable(8760, name='power from stu to off')
ele_stu2tch = cp.Variable(8760, name='power from stu to tch')
ele_off2tch = cp.Variable(8760, name='power from off to tch')
gas_purchased = cp.Variable(8760, name='purchased gas')

pv_max = N_pv * data_loader.supply_pv_8760
wt_max = N_wt * data_loader.supply_wt_8760

pv_stu_in = cp.Variable(8760, name='PV power in stu')
wt_off_in = cp.Variable(8760, name='WT power in off')

# =============================================================================================
# ================================= Create Constraints ========================================
# =============================================================================================
constraints = []
## 每条支路加入最大功率限制，容量为输入能量的最大值
## 学生区，支路功率约束
constraints += [
    #1 压缩式制冷机A输入功率限制
    V_stu_max[0,:] == cp.multiply(device_stu_max[0], hrs),
    V_stu_max[6,:] <= cp.multiply(device_stu_max[0], hrs),
    #2 电制气输入功率限制
    V_stu_max[1,:] == cp.multiply(device_stu_max[1], hrs),
    V_stu_max[7,:] <= cp.multiply(device_stu_max[1], hrs),
    #3 电锅炉输入功率限制
    V_stu_max[2,:] == cp.multiply(device_stu_max[2], hrs),
    V_stu_max[8,:] <= cp.multiply(device_stu_max[2], hrs),
    #4 热泵输入功率限制
    V_stu_max[3,:] == cp.multiply(device_stu_max[3], hrs),
    V_stu_max[9,:] <= cp.multiply(device_stu_max[3], hrs),
    #5 储电输入限制
    V_stu_max[5,:] == cp.multiply(device_stu_max[4], hrs),
    #5 储电输出限制 678910
    V_stu_max[6,:] <= cp.multiply(device_stu_max[4], hrs),
    V_stu_max[7,:] <= cp.multiply(device_stu_max[4], hrs),
    V_stu_max[8,:] <= cp.multiply(device_stu_max[4], hrs),
    V_stu_max[9,:] <= cp.multiply(device_stu_max[4], hrs),
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
    # 虚拟储能支路限制
    V_stu_max[22,:] == cp.multiply(device_stu_max[4], hrs),
    V_stu_max[23,:] == cp.multiply(device_stu_max[5], hrs),
    V_stu_max[24,:] == cp.multiply(device_stu_max[6], hrs),
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
    # 虚拟储能支路限制
    V_off_max[26,:] == cp.multiply(device_off_max[4], hrs),
    V_off_max[27,:] == cp.multiply(device_off_max[5], hrs),
    V_off_max[28,:] == cp.multiply(device_off_max[6], hrs),
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
    V_tch_max[17,:] <= cp.multiply(device_tch_max[1], hrs),
    #3 压缩式制冷机组B输入功率限制
    V_tch_max[2,:] == cp.multiply(device_tch_max[2], hrs),
    V_tch_max[6,:] == cp.multiply(device_tch_max[2], hrs),
    V_tch_max[10,:] == cp.multiply(device_tch_max[2], hrs),
    V_tch_max[14,:] == cp.multiply(device_tch_max[2], hrs),
    V_tch_max[18,:] <= cp.multiply(device_tch_max[2], hrs),
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
    V_tch_max[17,:] <= cp.multiply(device_tch_max[5], hrs),
    V_tch_max[18,:] <= cp.multiply(device_tch_max[5], hrs),
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
    # 虚拟储能支路限制
    V_tch_max[34,:] == cp.multiply(device_tch_max[5], hrs),
    V_tch_max[35,:] == cp.multiply(device_tch_max[6], hrs),
    V_tch_max[36,:] == cp.multiply(device_tch_max[7], hrs),
]

## 分区能源枢纽约束
constraints += [
    # 储能外支路功率约束
    VF_stu_min <= VF_stu, VF_off_min <= VF_off, VF_tch_min <= VF_tch,
    VF_stu <= VF_stu_max, VF_off <= VF_off_max, VF_tch <= VF_tch_max,
    VB_stu_min <= VB_stu, VB_off_min <= VB_off, VB_tch_min <= VB_tch,
    VB_stu <= VB_stu_max, VB_off <= VB_off_max, VB_tch <= VB_tch_max,

    # VB与VF之间的等式约束
    VB_stu == - QB_stu_inv @ QF_stu @ VF_stu - QB_stu_inv @ R_stu @ V_stu_in,
    VB_off == - QB_off_inv @ QF_off @ VF_off - QB_off_inv @ R_off @ V_off_in,
    VB_tch == - QB_tch_inv @ QF_tch @ VF_tch - QB_tch_inv @ R_tch @ V_tch_in,

    # 储能虚拟支路功率约束，注意这里充放电可正可负
    -V_stu_max[-3:] <= V_stu_storage, V_stu_storage <= V_stu_max[-3:],
    -V_off_max[-3:] <= V_off_storage, V_off_storage <= V_off_max[-3:],
    -V_tch_max[-3:] <= V_tch_storage, V_tch_storage <= V_tch_max[-3:],
]


## 负荷与输出约束
constraints += [
    # 电气热
    # 学生区输出拆分(4, 8760) 电气热冷
    V_gen_stu == V_gen_for_load_stu + V_gen_for_out_stu,
    0 <= V_gen_stu, 0 <= V_gen_for_load_stu, 0 <= V_gen_for_out_stu,
    V_gen_for_load_stu[1,:] == 0, # 本身没有气负荷
    V_gen_for_out_stu[2,:] == 0, # 没有热输出
    V_gen_for_out_stu[3,:] == 0, # 没有冷输出
    # 教学办公区输出拆分(3, 8760) 电热冷，没有产生气
    V_gen_off == V_gen_for_load_off + V_gen_for_out_off,
    0 <= V_gen_off, 0 <= V_gen_for_load_off, 0 <= V_gen_for_out_off,
    V_gen_for_out_off[2,:] == 0, # 没有冷输出
    # 教工区没有输出 (3, 8760) 电热冷，没有产生气
    V_gen_tch == V_gen_for_load_tch + V_gen_for_out_tch,
    0 <= V_gen_tch, 0 <= V_gen_for_load_tch, 0 <= V_gen_for_out_tch,
    V_gen_for_out_tch == 0, # 没有任何输出
    # 负荷约束
    V_gen_for_load_stu[[0,2,3],:] <= V_load_stu,
    V_gen_for_load_off <= V_load_off,
    V_gen_for_load_tch <= V_load_tch,
]


## 储能约束
for day in range(365):
    for hour in range(24):
        cur_hour = 24*day + hour
        constraints += [
            # SOC 上下限
            0 <= es_stu[:,cur_hour], es_stu[:,cur_hour]<= es_stu_max,
            0 <= es_off[:,cur_hour], es_off[:,cur_hour]<= es_off_max,
            0 <= es_tch[:,cur_hour], es_tch[:,cur_hour]<= es_tch_max,
        ]
        # 储能递推约束
        if hour == 23:
            constraints += [
                # 储能递推约束
                es_stu[:,cur_hour-23] == es_stu[:,cur_hour] + V_stu_storage[:,cur_hour],
                es_off[:,cur_hour-23] == es_off[:,cur_hour] + V_off_storage[:,cur_hour],
                es_tch[:,cur_hour-23] == es_tch[:,cur_hour] + V_tch_storage[:,cur_hour],
            ]
        else:
            constraints += [
                # 储能递推约束
                es_stu[:,cur_hour+1] == es_stu[:,cur_hour] + V_stu_storage[:,cur_hour],
                es_off[:,cur_hour+1] == es_off[:,cur_hour] + V_off_storage[:,cur_hour],
                es_tch[:,cur_hour+1] == es_tch[:,cur_hour] + V_tch_storage[:,cur_hour],
            ]

## 母线平衡
constraints += [
    # 风光约束
    0 <= pv_stu_in, pv_stu_in <= pv_max, 0 <= wt_off_in, wt_off_in <= wt_max, 
    # 购电、购气约束
    0 <= ele_stu_purchased, 0 <= ele_off_purchased, 0 <= ele_tch_purchased,
    0 <= ele_purchased, 0 <= gas_purchased,
    ## 母线平衡
    # V_stu_in 电热输入+3储能， V_stu_out 电气热冷
    # V_off_in 电气输入+3储能， V_off_out 电热冷
    # V_tch_in 电气热输入+3储能，V_tch_out 电热冷

    # 电母线平衡
    ele_stu_purchased + V_gen_for_out_stu[0,:] + pv_stu_in == V_stu_in[0,:] + ele_stu2off + ele_stu2tch,
    ele_off_purchased + ele_stu2off + V_gen_for_out_off[0,:] + wt_off_in == V_off_in[0,:] + ele_off2tch,
    ele_tch_purchased + ele_stu2tch + ele_off2tch == V_tch_in[0,:],
    ele_purchased == ele_stu_purchased + ele_off_purchased + ele_tch_purchased,
    # 气母线平衡
    gas_purchased + V_gen_for_out_stu[1,:] == V_off_in[1,:] + V_tch_in[1,:],
    # 热母线平衡
    V_gen_for_out_off[1,:] == V_stu_in[1,:] + V_tch_in[2,:] 
]

# =============================================================================================
# ===================================== object funtion ========================================
# =============================================================================================
# C_total = C_cap + C_op + C_carbon

## C_cap 建设成本
C_cap_stu, _ = std_model_stu.compute_C_cap(N_stu, N_pv)
C_cap_off, _ = std_model_off.compute_C_cap(N_off, N_wt)
C_cap_tch, _ = std_model_tch.compute_C_cap(N_tch)

C_cap = C_cap_stu + C_cap_off + C_cap_tch

## C_op 运行成本
unmet_stu = cp.sum(V_load_stu - V_gen_for_load_stu[[0,2,3],:])
unmet_off = cp.sum(V_load_off - V_gen_for_load_off)
unmet_tch = cp.sum(V_load_tch - V_gen_for_load_tch)
unmet = unmet_stu + unmet_off + unmet_tch # 总失负荷 MWh

C_unmet = UNMET_PENALTY * unmet
C_ele = data_loader.price_electricity_8760 @ ele_purchased
C_gas = data_loader.price_gas_8760 @ gas_purchased * GAS2MWH
C_op = C_ele + C_gas + C_unmet

## C_carbon 碳排放成本
carbon = ele_purchased @ data_loader.carbon_factor_8760 + gas_purchased * GAS2MWH * CARBON_PER_GAS # tCO2
carbon_exceed = cp.Variable()
constraints += [
    carbon_exceed >= cp.sum(carbon) - CARBON_MAX,
    carbon_exceed >= 0,
]
C_carbon = CARBON_PENALTY * carbon_exceed

## C_total 最终目标函数
C_total = C_cap + C_op + C_carbon

if __name__ == '__main__':
    obj = cp.Minimize(C_total)
    problem = cp.Problem(obj, constraints)
    # 设置Gurobi的method参数
    solver_options = {"Method": 3}  
    # -1 表示默认方法，3 表示非确定性并发，4 表示确定性并发

    # 解决问题
    problem.solve(verbose=True, solver=cp.GUROBI, solver_opts=solver_options)
    # problem.solve(verbose=True, solver=cp.GUROBI)

    status = problem.status
    print(f"Problem Status: {status}")

    print(f"The optimal value is {problem.value}")

    print(f"C_total {C_total.value} = C_cap {C_cap.value} + C_op {C_op.value} + C_carbon {C_carbon.value}")
    print(f"C_cap {C_cap.value} = C_cap_stu {C_cap_stu.value} + C_cap_off {C_cap_off.value} + C_cap_tch {C_cap_tch.value}")
    print(f"C_op {C_op.value} = C_ele {C_ele.value} + C_gas {C_gas.value} + C_unmet {C_unmet.value}")

    print(f"总失负荷： {unmet.value}")

    print(f"C_carbon {C_carbon.value} = CARBON_PENALTY {CARBON_PENALTY} * carbon_exceed {carbon_exceed.value}")

    print(f"总碳排放：{sum(carbon.value)}, 超限 {carbon_exceed.value}")

    V_stu, V_off, V_tch = V_stu.value, V_off.value, V_tch.value
    N_stu, N_off, N_tch, N_pv, N_wt = N_stu.value, N_off.value, N_tch.value, N_pv.value, N_wt.value

    save_N(N_stu, N_off, N_tch, N_pv, N_wt)
    save_V(V_stu, V_off, V_tch)



