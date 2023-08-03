import math

from Chiller import Chiller_Model
import numpy as np
# from sympy import *

class model:
    #接受系统的总的冷负荷
    def __init__(self):
        self.index = 0
        self.CLs = np.load("env_1h(test).npy")
        self.c_p=4.2
        self.density_water =1000
        self.T_chwr_ref = 17
        self.P_chiller_ref = 314
    def step(self,action):
        chiller = Chiller_Model(self.CLs[self.index],action)
        #返回大小冷机的冷机功率、水泵功率、冷却塔风机功率
        CLc,T_chwr,Chiller_number,T_outdoor = chiller.get_variable()
        P_chiller, P_tower, P_pump, self.T_chwr, Tcwr_big, self.T_cws = chiller.get_P()
        P_total = P_chiller+ P_tower +P_pump
        P_total_ref = 419  #(314+75+30)
        #计算奖赏
        R_p = 1-P_total/P_total_ref
        R_c = 1 / (1+ 0.25* math.exp( 5 * (T_chwr - self.T_chwr_ref)))
        R = 0.6* R_p + 0.4 * R_c
        self.index += 1
        Done = False
        if self.index >= len(self.CLs)-1:
            Done = True
        S_ = self.CLs[self.index]
        return S_,CLc,Done,P_chiller,P_tower,R,R_p,R_c,T_chwr,T_outdoor,P_total,Chiller_number
    def reset(self):
        self.index = 0
        return self.CLs[self.index]
if __name__=="__main__":
    pass