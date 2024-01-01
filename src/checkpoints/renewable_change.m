% 这个程序用于将风电光伏数据改为你需要的数值
% change the solar and wind ouput as your planning results
% and put the dealed data in CLOUDGOPT simulation platform 

%% 读取数据
[num,txt]   =   xlsread("【数据】出力曲线_风电.xlsx");
[num2,txt2] =   xlsread("【数据】出力曲线_光伏.xlsx");
%% 输出数据
wind_MW  = 64;% 需要乘以的倍数  installed capacity
solar_MW = 469;% 需要乘以的倍数  installed capacity


%% 输出数据
xlswrite("【数据】出力曲线_风电_MOD.xlsx",txt,'sheet1');
xlswrite("【数据】出力曲线_风电_MOD.xlsx",[num(:,1),num(:,2:25)*wind_MW],'sheet1','A2');

xlswrite("【数据】出力曲线_光伏_MOD.xlsx",txt2,'sheet1');
xlswrite("【数据】出力曲线_光伏_MOD.xlsx",[num2(:,1),num2(:,2:25)*solar_MW],'sheet1','A2');
