import numpy as np


def rref(A_, tol=1.0e-12):
    A = A_.copy()
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb

save_dir = './checkpoints/'

def save_N(N_stu, N_off, N_tch, N_pv, N_wt):
    np.save(save_dir+'N_stu.npy', N_stu)
    np.save(save_dir+'N_off.npy', N_off)
    np.save(save_dir+'N_tch.npy', N_tch)
    np.save(save_dir+'N_pv.npy', N_pv)
    np.save(save_dir+'N_wt.npy', N_wt)

def load_N():
    N_stu = np.load(save_dir+'N_stu.npy')
    N_off = np.load(save_dir+'N_off.npy')
    N_tch = np.load(save_dir+'N_tch.npy')
    N_pv = np.load(save_dir+'N_pv.npy')
    N_wt = np.load(save_dir+'N_wt.npy')
    print(f"N_stu: {N_stu}")
    print(f"N_off: {N_off}")
    print(f"N_tch: {N_tch}")
    print(f"N_pv: {N_pv}")
    print(f"N_wt: {N_wt}")
    return N_stu, N_off, N_tch, N_pv, N_wt

def save_V(V_stu, V_off, V_tch):
    np.save(save_dir+'V_stu.npy', V_stu)
    np.save(save_dir+'V_off.npy', V_off)
    np.save(save_dir+'V_tch.npy', V_tch)

def load_V():
    V_stu = np.load(save_dir+'V_stu.npy')
    V_off = np.load(save_dir+'V_off.npy')
    V_tch = np.load(save_dir+'V_tch.npy')
    print(f"V_stu: {V_stu}")
    print(f"V_off: {V_off}")
    print(f"V_tch: {V_tch}")
    return V_stu, V_off, V_tch

import openpyxl

def multiply_excel_values(input_file, output_file, factor):
    # 打开源文件
    wb = openpyxl.load_workbook(input_file)
    
    # 选择第一个工作表
    sheet = wb.active
    
    # 循环遍历所有的行和列
    for row in sheet.iter_rows():
        for idx, cell in enumerate(row):
            # 如果是第一列，跳过
            if idx == 0:
                continue
            # 如果单元格包含数字，则将其乘以指定的因子
            if cell.value and isinstance(cell.value, (int, float)):
                cell.value *= factor
    
    # 保存更改到新文件
    wb.save(output_file)

# test
