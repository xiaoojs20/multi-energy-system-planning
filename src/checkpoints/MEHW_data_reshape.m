% 这个程序用于将MEOS导出的数据转化为OJ平台可以处理的数据


%% 读取数据
[num,txt]   =   xlsread("平台导出文件2.xls");


%% 进行数据的转写
% LOAD_CUR={"能量枢纽切负荷量"};

ele_cur_idx = find(strcmp(txt(:,3),'能量枢纽切负荷量（MW）')&contains(txt(:,5),'电负荷')) ;
wam_cur_idx = find(strcmp(txt(:,3),'能量枢纽切负荷量（MW）')&contains(txt(:,5),'热负荷')) ;
col_cur_idx = find(strcmp(txt(:,3),'能量枢纽切负荷量（MW）')&contains(txt(:,5),'冷负荷')) ;

MAT_thermal_source  =   num(find(strcmp(txt(:,3),'火电出力（MW）'))-1,8:31);
MAT_gas_source      =   num(find(strcmp(txt(:,3),'气源出力（MW）'))-1,8:31);
MAT_planning_result =   num(find(num(:,2)>200),8);

%% 进行数据的输出

ans_load1           =   sum(reshape(num(ele_cur_idx-1, 8:31)',[],3),2);
ans_load2           =   sum(reshape(num(wam_cur_idx-1, 8:31)',[],3),2);
ans_load3           =   sum(reshape(num(col_cur_idx-1, 8:31)',[],3),2);
% 默认在枢纽中有三个负荷节点
ans_ele             =   sum(reshape(MAT_thermal_source',[],3),2);
ans_gas             =   reshape(MAT_gas_source',[],1);
ans_planning        =   zeros(8760,1);
ans_planning(1:18)  =   reshape(MAT_planning_result',[],1);


result_table=table(ans_load1,ans_load2,ans_load3, ans_ele, ans_gas, ans_planning);
writetable(result_table, 'HWdata_for_OJ.csv');
%csvwrite('HWdata_for_OJ.csv',[ARRAY_load_cur ARRAY_thermal_source ans_gas ARRAY_planning_result]);
disp('数据转写完成！你的失负荷量为：')
disp(num2str(sum(ans_load1+ans_load2+ans_load3)))