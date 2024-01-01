from utils import load_N, multiply_excel_values
from std_modeling import StdModel

std_model_stu, std_model_off, std_model_tch = StdModel(path='../data', region='stu'), StdModel(path='../data', region='off'), StdModel(path='../data', region='tch')

N_stu, N_off, N_tch, N_pv, N_wt = load_N()


C_cap_stu, C_cap_stu_list = std_model_stu.compute_C_cap(N_stu, N_pv)
C_cap_off, C_cap_off_list = std_model_off.compute_C_cap(N_off, N_wt)
C_cap_tch, C_cap_tch_list = std_model_tch.compute_C_cap(N_tch)

C_cap = C_cap_stu + C_cap_off + C_cap_tch

print(f"C_cap {C_cap} = C_cap_stu {C_cap_stu} + C_cap_off {C_cap_off} + C_cap_tch {C_cap_tch}")



# 使用示例，将 "input.xlsx" 中的每个数字乘以2，并保存为 "output.xlsx"
input_file = "../data/【数据】出力曲线_光伏.xlsx"
output_file = "../data/【数据】出力曲线_光伏_修改后.xlsx"

multiply_excel_values(input_file, output_file, 276)

input_file = "../data/【数据】出力曲线_风电.xlsx"
output_file = "../data/【数据】出力曲线_风电_修改后.xlsx"

multiply_excel_values(input_file, output_file, 262)


print(f"############################################ 学生区 总成本 = {C_cap_stu} ##############################")
for i in range(8):
    if i < 7:
        print(f"{std_model_stu.devic_name[i]:<15}\t个数 = {N_stu[i]:<10}\t总功率容量 = {N_stu[i]*std_model_stu.Pbase[i]} MW or MWh\
              \t总成本 = {C_cap_stu_list[i]}")
    else:
        print(f"{std_model_stu.devic_name[-1]:<15}\t个数 = {N_pv[0]:<10}\t总功率容量 = {N_stu[-1]*std_model_stu.Pbase[-1]}MW or MWh\
              \t总成本 = {C_cap_stu_list[i]}")
print(f"########################################## 教学办公区 总成本 = {C_cap_off} #############################")
for i in range(8):
    if i < 7:
        print(f"{std_model_off.devic_name[i]:<15}\t个数 = {N_off[i]:<10}\t总功率容量 = {N_off[i]*std_model_off.Pbase[i]}MW or MWh\
              \t总成本 = {C_cap_off_list[i]}")
    else:
        print(f"{std_model_off.devic_name[-1]:<15}\t个数 = {N_wt[0]:<10}\t总功率容量 = {N_off[-1]*std_model_off.Pbase[-1]}MW or MWh\
              \t总成本 = {C_cap_off_list[i]}")
print(f"############################################ 教工区 总成本 = {C_cap_tch} ##############################")
for i in range(8):
    print(f"{std_model_tch.devic_name[i]:<15}\t个数 = {N_tch[i]:<10}\t总功率容量 = {N_tch[i]*std_model_tch.Pbase[i]}MW or MWh\
              \t总成本 = {C_cap_tch_list[i]}")