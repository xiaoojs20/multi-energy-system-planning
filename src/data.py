import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

class EnergyData(object):
    """
    supply: electricity. wind and pv (unit: MW)
    load: heating, cooling and electricity (unit MW)
    price: electricity (unit: rmb/MWh) and gas (unit: rmb/m3)
    carbon factor: tCO2/m3
    """
    def __init__(self, path):
        # path = '../data/' maybe.
        # initialize the dataframe
        self.supply_wind = pd.read_excel(os.path.join(path, 'supply_wind.xlsx'))
        self.supply_pv = pd.read_excel(os.path.join(path, 'supply_pv.xlsx'))

        self.load_heating_jg = pd.read_excel(os.path.join(path, 'load_heating_jg.xlsx'))
        self.load_heating_jxbg = pd.read_excel(os.path.join(path, 'load_heating_jxbg.xlsx'))
        self.load_heating_xs = pd.read_excel(os.path.join(path, 'load_heating_xs.xlsx'))

        self.load_cooling_jg = pd.read_excel(os.path.join(path, 'load_cooling_jg.xlsx'))
        self.load_cooling_jxbg = pd.read_excel(os.path.join(path, 'load_cooling_jxbg.xlsx'))
        self.load_cooling_xs = pd.read_excel(os.path.join(path, 'load_cooling_xs.xlsx'))

        self.load_electricity_jg = pd.read_excel(os.path.join(path, 'load_electricity_jg.xlsx'))
        self.load_electricity_jxbg = pd.read_excel(os.path.join(path, 'load_electricity_jxbg.xlsx'))
        self.load_electricity_xs = pd.read_excel(os.path.join(path, 'load_electricity_xs.xlsx'))

        self.price_electricity = pd.read_excel(os.path.join(path, 'price_electricity.xlsx'))
        self.price_gas = pd.read_excel(os.path.join(path, 'price_gas.xlsx'))
        # print(self.price_gas)

        self.carbon_factor = pd.read_excel(os.path.join(path, 'carbon_factor.xlsx'), header=None)
        # print(self.carbon_factor)
        
    def get_2d_array(self):
        self.supply_wind_2d_array = self.supply_wind.iloc[:,1:].values
        self.supply_pv_2d_array = self.supply_pv.iloc[:,1:].values

        self.load_heating_jg_2d_array = self.load_heating_jg.iloc[:,1:].values
        self.load_heating_jxbg_2d_array = self.load_heating_jxbg.iloc[:,1:].values
        self.load_heating_xs_2d_array = self.load_heating_xs.iloc[:,1:].values
        self.load_cooling_jg_2d_array = self.load_cooling_jg.iloc[:,1:].values
        self.load_cooling_jxbg_2d_array = self.load_cooling_jxbg.iloc[:,1:].values
        self.load_cooling_xs_2d_array = self.load_cooling_xs.iloc[:,1:].values
        self.load_electricity_jg_2d_array = self.load_electricity_jg.iloc[:,1:].values
        self.load_electricity_jxbg_2d_array = self.load_electricity_jxbg.iloc[:,1:].values
        self.load_electricity_xs_2d_array = self.load_electricity_xs.iloc[:,1:].values

        self.price_electricity_2d_array = self.price_electricity.iloc[:,1:].values
        self.price_gas_2d_array = self.price_gas.iloc[:,1:].values
        self.carbon_factor_2d_array = self.carbon_factor.iloc[:,1:].values
        return self

    def get_8760(self):
        self.supply_wt_8760 = self.supply_wind_2d_array.flatten()
        self.supply_pv_8760 = self.supply_pv_2d_array.flatten()

        self.load_heating_jg_8760 = self.load_heating_jg_2d_array.flatten()
        self.load_heating_jxbg_8760 = self.load_heating_jxbg_2d_array.flatten()
        self.load_heating_xs_8760 = self.load_heating_xs_2d_array.flatten()
        self.load_cooling_jg_8760 = self.load_cooling_jg_2d_array.flatten()
        self.load_cooling_jxbg_8760 = self.load_cooling_jxbg_2d_array.flatten()
        self.load_cooling_xs_8760 = self.load_cooling_xs_2d_array.flatten()
        self.load_electricity_jg_8760 = self.load_electricity_jg_2d_array.flatten()
        self.load_electricity_jxbg_8760 = self.load_electricity_jxbg_2d_array.flatten()
        self.load_electricity_xs_8760 = self.load_electricity_xs_2d_array.flatten()

        self.price_electricity_8760 = self.price_electricity_2d_array.flatten()
        self.price_gas_8760 = self.price_gas_2d_array.flatten()
        self.carbon_factor_8760 = self.carbon_factor_2d_array.flatten()
        self.penalty_unmet_8760 = np.full(8760, 500000)
        return self


class Device(object):
    """
    name, eta, cost, ttl, Pbase, Ebase
    """
    def __init__(self, path, device_name):
        df_device = pd.read_csv(os.path.join(path, 'device.csv'))
        # print(df_device)
        if device_name in df_device['name'].values:
            indices = np.where(df_device['name'].values == device_name)[0][0]
            # print(indices)
            self.name = device_name
            self.cost = df_device['cost'].iloc[indices]
            self.ttl = df_device['ttl'].iloc[indices]
            self.Pbase = df_device['Pbase'].iloc[indices]
            self.Ebase = df_device['Ebase'].iloc[indices]
        print (self.cost)

if __name__ == '__main__':
    # data_loader = EnergyData(path="../data")
    # data_loader = data_loader.get_2d_array().get_8760() 
    # print(data_loader.price_electricity_2d_array.shape)
    # print(type(data_loader.price_electricity_2d_array))
    # print(data_loader.price_electricity_8760.shape)
    # print(type(data_loader.price_electricity_8760))

    # print(wind_list[0][0])

    # hours = np.arange(8760)
    ## heating load 
    # plt.figure(figsize=(12, 6))
    # plt.plot(hours, data_loader.load_heating_xs_8760, label='student')
    # plt.plot(hours, data_loader.load_heating_jxbg_8760, label='teaching & office')
    # plt.plot(hours, data_loader.load_heating_jg_8760, label='teacher')
    # plt.xlabel('Hour (h)')
    # plt.ylabel('Demand (MW)')
    # plt.title('Heating Load')
    # plt.legend()
    # plt.show()

    ## plot load
    # heating_load = data_loader.load_heating_xs_8760 + data_loader.load_heating_jg_8760 + data_loader.load_heating_jxbg_8760
    # cooling_load = data_loader.load_cooling_xs_8760 + data_loader.load_cooling_jg_8760 + data_loader.load_cooling_jxbg_8760
    # electricity_load = data_loader.load_electricity_xs_8760 + data_loader.load_electricity_jg_8760 + data_loader.load_electricity_jxbg_8760
    # plt.figure(figsize=(12, 6))
    # plt.plot(hours, heating_load, label='heating', color='r')
    # plt.plot(hours, cooling_load, label='cooling', color='b')
    # plt.plot(hours, electricity_load, label='electriciy', color='y')
    # plt.xlabel('Hour (h)')
    # plt.ylabel('Demand (MW)')
    # plt.title('Year-round Load')
    # plt.legend()
    # plt.savefig('../fig/load.png')

    ########################################## device 
    CERS_A = Device(path="../data", device_name='压缩式制冷机组A')
