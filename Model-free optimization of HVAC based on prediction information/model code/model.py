import math

from Chiller import Chiller_Model
import numpy as np
from sympy import *

class model:
    #接受系统的总的冷负荷
    def __init__(self):
        self.index = 0
        self.CLs = np.load("env_1h.npy")
        self.c_p=4.2
        self.density_water =1000
        self.F_chw = 252
        self.T_chwr_ref = 17
        self.P_chiller_ref = 314
    def step(self,action):
        chiller = Chiller_Model(self.CLs[self.index],action)
        #返回大小冷机的冷机功率、水泵功率、冷却塔风机功率
        CLc,T_chwr,Chiller_number,T_outdoor = chiller.get_variable()
        P_chiller= chiller.caculate_P_chiller()
        P_chiller_total = P_chiller * Chiller_number
        k1 = 0.6
        K2 = 0.4
        #计算奖赏
        Es = 1- P_chiller/self.P_chiller_ref
        comfort = 1/(1+0.25*math.exp( 5*T_chwr-self.T_chwr_ref ))
        R =k1*Es + K2*comfort
        self.index += 1
        Done = False
        if self.index >= len(self.CLs)-1:
            Done = True
        S_ = self.CLs[self.index]
        return S_,CLc,Done,P_chiller,R,Es,comfort,T_chwr,T_outdoor,P_chiller_total
    def reset(self):
        self.index = 0
        return self.CLs[self.index]
if __name__=="__main__":
    pass