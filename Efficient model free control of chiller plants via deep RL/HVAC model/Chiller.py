import numpy as np
import math
'''
模型的参数列表
Approach:临近温度，出口水温-进口空气湿球温度 
CL_s :系统的冷却负荷,楼每时刻需要的制冷量
CL_c: 冷机冷负荷，计算方式：单台冷冻机的进出口温差*冷冻水流量
c_p:水的比热容，常量为4200 J/(kg·K)
f_nominal:水泵名义频率（50HZ） 
f_pump:泵的工作频率
density_water:水的密度 常量为：1000 kg/m3
F_chw: 冷冻水流量,可以直接指定，该值的确定方式和用户侧阻抗有关，裘老大推荐直接按照冷机的名义冷冻水流量来定。
F_chiller:冷机的额定制冷量
F_nominal:水泵名义流量
PLR:冷机负荷率
P_ref:额定功率，和设备有关
T_bt:bulb temperature
T_chws:冷冻水出水温度（冷水机出口），（该值由控制器直接指定）
T_chwr:冷冻水回水温度（冷水机进口）
T_cwr:冷却水回水温度
T_wb:进气湿球-灯泡温度  
T_wet:室外的湿球温度
PLR=Clc/冷机的额定制冷量(0,1) 
T_chws:冷冻水出水温度（冷水机出口），（该值由控制器直接指定）
F_chw:冷冻水流量,可以直接指定，该值的确定方式和用户侧阻抗有关，裘老大推荐直接按照冷机的名义冷冻水流量来定。
P_ref:额定功率  #159.7
F_chiller:冷机的额定制冷量
'''

class Chiller_Model:
    def __init__(self,CLS,action):
        self.CLS = CLS[0]
        self.T_outdoor = CLS[1]
        self.T_cwr = CLS[1]
        self.T_chws = action[0]
        self.f_tower = action [1]
        self.T_chwr = self.get_variable()[1]
        self.CLC = self.get_variable()[0]
        self.chiller_number = self.get_variable()[3]
        self.coeff = np.load("coeff_Date.npy")
        self.step_len = 0.2
        self.f_nominal = 50
        self.F_nominal = 581
        self.F_cw = 1000
        self.F_tower = 569
        self.F_chiller = 1760  #冷机的额定制冷量
        self.c_p = 4.2    #水的比热容
        self.density_water  = 1000 #水的密度


    def get_variable(self):

        Chiller_number = math.ceil(self.CLS / 1760)
        CLc = self.CLS / Chiller_number
        c_p = 4.2
        density_water = 1000
        F_chw = 252
        T_outdoor = self.T_outdoor
        T_chwr = self.T_chws + CLc / ((c_p * density_water * (F_chw ) / 3600))
        return CLc,T_chwr,Chiller_number,T_outdoor

    def get_P(self):
            f_pump = 50
            f_tower = self.f_tower
            is_Converge_Data_T_cwr = []
            #进行循环来使得Tcwr(冷却水的温度变得恒定)
            step = 0
            P_chiller = 0
            Tcwr_big = 0
            #建立该循环是为了使得整个Tcws或者Tcwr能够收敛到一个具体的值。
            while True:
                #+++++++++++++++++冷水机模型++++++++++++++++++++++
                self.T_chwr = self.get_variable()[1]
                # 开始计算P_chillier值，根据公式1,这边使用了T_cwr
                P_chiller = self.caculate_P_chiller()
                # 计算T_cws
                self.T_cws = self.T_cwr + (P_chiller + self.CLC)/((self.c_p * self.F_cw * self.density_water) / (3600))
                #+++++++++++++++++冷却水泵模型++++++++++++++++++++++
                # 下面开始计算冷却水泵模型，计算冷却水流量
                self.F_cw = ((f_pump) / (self.f_nominal)) * self.F_nominal
                # print("self.F_cw:",self.F_cw)
                #+++++++++++++++++冷却塔模型++++++++++++++++++++++++
                # 计算出Error值
                minerror,T_cwr = self.caculate_Error(f_tower,f_pump)
                self.T_cwr = T_cwr
                #判断T_cwr这个值是不是收敛了。
                is_Converge_Data_T_cwr.append(self.T_cwr)
                #收敛条件
                if (len(is_Converge_Data_T_cwr) >= 2 and abs(is_Converge_Data_T_cwr[step] - is_Converge_Data_T_cwr[step-1])<=0.1) or step>=50:
                    break
                step += 1
            P_pump = 75
            P_tower = self.caculate_P_tower(f_tower) * (581 / 240)
            #根据开关状态来
            return P_chiller,P_tower,P_pump,self.T_chwr,Tcwr_big,self.T_cws


    def caculate_Error(self,f_tower,f_pump):
        FRair = f_tower/50
        FRwater = ((f_pump / 50)*self.F_nominal)/self.F_tower
        T_wb = self.T_outdoor
        T_cwr = T_wb
        T_cws = self.T_cws
        ERRORs = []
        all_T_cwr = []
        for T_cwr_times in range(int(50/self.step_len)):
            T_cwr += self.step_len
            Tr = T_cws - T_cwr
            Approach = self.coeff[0]+self.coeff[1]*FRair+self.coeff[2]*(FRair ** 2)+self.coeff[3]*(FRair ** 3)+self.coeff[4]*FRwater+self.coeff[5]*FRair*FRwater\
                       +self.coeff[6]  * (FRair**2)*FRwater+self.coeff[7]*(FRwater**2)+self.coeff[8]*FRair*(FRwater ** 2)+self.coeff[9]*(FRwater ** 3)+self.coeff[10]*T_wb+self.coeff[11]*FRair*T_wb\
                       +self.coeff[12] * (FRair ** 2)* T_wb + self.coeff[13] * FRwater * T_wb + self.coeff[14] * FRwater * T_wb * FRair + self.coeff[15]*(FRwater**2)*T_wb\
                       +self.coeff[16] * (T_wb ** 2) + self.coeff[17] * FRair * (T_wb ** 2) + self.coeff[18]*FRwater*(T_wb**2)\
                       +self.coeff[19] * (T_wb ** 3) + self.coeff[20] * Tr + self.coeff[21] * FRair * Tr + self.coeff[22]*Tr*(FRair**2)\
                       +self.coeff[23] * Tr * FRwater+ self.coeff[24] * FRwater * FRair * Tr + self.coeff[25]*Tr*(FRwater**2)+\
                       +self.coeff[26] * Tr * T_wb + self.coeff[27] * FRair * T_wb * Tr + self.coeff[28]*FRwater*T_wb*Tr+self.coeff[29]*(T_wb**2)*Tr\
                       +self.coeff[30] * (Tr ** 2) + self.coeff[31] * FRair * (Tr ** 2) + self.coeff[32]*FRwater*(Tr**2)\
                       +self.coeff[33] * T_wb * (Tr ** 2) + self.coeff[34] * (Tr ** 3)
            T_cwr_y = Approach + T_wb
            Error = abs(T_cwr - T_cwr_y)
            ERRORs.append(Error)
            all_T_cwr.append(T_cwr)
        ERRORs = np.array(ERRORs)
        all_T_cwr = np.array(all_T_cwr)
        return np.min(ERRORs),all_T_cwr[np.argmin(ERRORs)]

    def caculate_P_chiller(self):     # 计算冷机的功率 一个与三个参数相关 T_chws ,T_cwr，plr
        b1 = 4.575085E-01
        b2 = 1.313508E-01
        b3 = -4.408831E-03
        b4 = 1.930354E-02
        b5 = -5.479641E-04
        b6 = -1.376580E-03
        d1 = 6.794525E-01
        d2 = 6.694756E-02
        d3 = -3.625396E-03
        d4 = -1.018762E-02
        d5 = 1.066394E-03
        d6 = -2.113402E-03
        g1 = 7.859908E-02
        g2 = 1.950291E-01
        g3 = 7.241581E-01
        F_chiller = 1760
        P_ref = 314
        PLR = self.CLC / F_chiller
        chillerCapFTemp = b1 + b2 * self.T_chws + b3 * self.T_chws ** 2 + b4 * self.T_cwr + b5 * self.T_cwr ** 2 + b6 * self.T_cwr * self.T_chws
        chillerEIREFTemp = d1 + d2 * self.T_chws + d3 * self.T_chws ** 2 + d4 * self.T_cwr + d5 * self.T_cwr ** 2 + d6 * self.T_cwr * self.T_chws
        chillerRIRFPLR = g1 + g2 * PLR + g3 * PLR ** 2
        P_chiller = P_ref * chillerCapFTemp * chillerRIRFPLR * chillerEIREFTemp
        return P_chiller

    def caculate_P_tower(self,f_tower):
        p_tower_a = 0.00684
        p_tower_b = -0.31463
        p_tower_c = 5.221
        self.P_tower = p_tower_a * f_tower**2 + p_tower_b*f_tower +p_tower_c
        return self.P_tower

if __name__ == "__main__":
    pass
