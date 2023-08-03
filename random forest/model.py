from numpy import array
from pandas import DataFrame, read_csv
from equipment_model import Chiller, Cooling_tower, Cooling_pump
import math
import numpy as np

# define rated cooling capacity, rated chilled supply water, rated chilled and cooling water flow rate, rated chilled water pump power
Q0, MIN_FACTOR, TCHWS_SET, FCHW_SET, Q_CHW = 1200, 0.2, 8, 150, 12

class chiller_plants:
    def __init__(self, env_df=DataFrame()):
        self.s = np.load("env_1h.npy")
        self.index = 0
        self.chiller = None
        self.pump = None
        self.tower = None
        self.counter = 0
        self.T_chwr_ref = 17
        self.Tchws = 8
        self.Tchwr = 12
        self.Tcws = 35
        self.Tcwr = 30
        self.Fcw = 480
        self.Fchws = 150
        self.Pchiller_ref = 159.7
        self.Ptower_ref = 7.5
    
    def step(self, action):

        self.chiller = Chiller(1060, 'model/BL/chiller_model.pkl', 4)
        pump_df = read_csv('model/BL/pump_model.csv', index_col=0)
        self.pump = Cooling_pump([], pump_df.loc['P'].values, 50, 4)
        tower_df = read_csv('model/BL/tower_model.csv', index_col=0)
        self.tower = Cooling_tower('model/BL/tower_model.pkl', tower_df.loc['power'].values, 50, 260, 4)

        TCHWS_SET = action[0]
        t_f = action[1]
        Twb, CL = self.s[self.index][1],self.s[self.index][0]

        if CL <= MIN_FACTOR*Q0:
            # chiller turn off
            self.chiller.Q, self.chiller.Tchws, self.chiller.number = 0, 0, 0
        elif CL <= 4*Q0:
            self.chiller.number = CL//Q0 + 1
            self.chiller.Q, self.chiller.Tchws = CL/self.chiller.number, TCHWS_SET
        else:
            self.chiller.number, self.chiller.Q = 4, Q0
            self.Tchws = TCHWS_SET
            self.chiller.Tchws = self.Tchws

        # calculate Tchws and assign it for each chiller
        self.Tchwr = self.Tchws+ CL / ((4.2 * 1000 * self.Fchws) / 3600)
        self.chiller.Tchwr = self.Tchwr
        # initialize equipment class and calculate cooling water flow rate
        self.tower.f, self.tower.number, self.tower.Twb = t_f, self.chiller.number, Twb
        self.Fcw = 240
        self.pump.Fcw, self.tower.Fcw = 240, 260
        # ierations to calculate Tcwr
        Tcwr_assume, iter_number = self.Tcwr, 0
        while True:
            Q_heatechange = self.chiller.calculate_Qct()*self.chiller.number
            Tcws_calculate = Tcwr_assume + 3.6/4.192*Q_heatechange/self.Fcw
            self.tower.Tcws = Tcws_calculate
            Tcwr_calculate = self.tower.calculate_Tcwr()
            if abs(Tcwr_assume-Tcwr_calculate) < 0.1:
                self.Tcwr, self.Tcws = Tcwr_assume, Tcws_calculate
                break
            else:
                if Tcwr_assume > Tcwr_calculate:
                    Tcwr_assume = Tcwr_assume-0.1
                else:
                    Tcwr_assume = Tcwr_assume+0.1
            if iter_number == 50:
                self.Tcwr, self.Tcws = Tcwr_assume, Tcws_calculate
                break
            iter_number += 1
        # calculate reward
        Tchwr = self.Tchwr
        Tcwr = self.Tcwr
        Pchiller = (self.chiller.calculate_P()+Q_CHW) *self.chiller.number
        Ptower = self.tower.calculate_P()
        Ppump = self.pump.calculate_P()
        Ptotal = Pchiller+Ptower+Ppump
        P_total = 1- ((self.chiller.calculate_P()+Q_CHW)+self.tower.calculate_P()) /((self.Pchiller_ref+Q_CHW)+self.Ptower_ref * self.chiller.number)
        comfort =  1 / (1+0.25 * math.exp(5 *(self.Tchwr- self.T_chwr_ref )))
        R = 0.6 * P_total + 0.4 * comfort
        self.index += 1
        Done = False
        if self.index >= len(self.s)-1:
            Done = True
        s_ = self.s[self.index]
        # return S_, reward and intermediate variable
        return s_,R,Pchiller,Ptower,TCHWS_SET,t_f,Ptotal,Tchwr,Tcwr,Done

    def reset(self):
        self.index = 0
        return self.s[self.index]

if __name__ == '__main__':
    pass