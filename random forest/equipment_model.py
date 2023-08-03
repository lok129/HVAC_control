from joblib import load
from numpy import array, arange, ones
from pandas import DataFrame, read_csv


class Chiller:
    
    def __init__(self, Q0, model_path, number,):
        self.Q0 = Q0
        self.model_path = model_path
        self.number = number
        self.Tchwr = 17
        self.Tchws = 10
        self.Tcwr = 30.5
        self.Fchw = 131
        self.Q = 1060

    def calculate_COP(self):
        if self.Q != 0:
            lgb_reg = load(self.model_path)
            [COP] = lgb_reg.predict(array([[self.Q, self.Tcwr, self.Tchws, self.Fchw]]))
            return COP
        else:
            return 0

    def calculate_P(self):
        if self.Q != 0:
            P = self.Q / self.calculate_COP()
            return P
        else:
            return 0

    def calculate_Qct(self):
        if self.Q != 0:
            Qct = self.Q * (1+1/self.calculate_COP())
            return Qct
        else:
            return 0
        

class Cooling_pump:
    
    def __init__(self, H_coefficients, P_coefficients, f0, number):
        self.H_coefficients = H_coefficients
        self.P_coefficients = P_coefficients
        self.f0 = f0
        self.number = number
        self.f = 40
        self.Fcw = 240
        
    def calculate_H(self):
        kair = self.f / self.f0
        H = self.H_coefficients[1]*(self.Fcw/self.number)**2+self.H_coefficients[2]*(self.Fcw/self.number)*kair+self.H_coefficients[3]*kair**2
        return H

    def calculate_S(self):
        H = self.calculate_H()
        S = H / (self.Fcw*self.number)**2
        return S

    def calculate_P(self):
        kair = self.f / self.f0
        P = self.P_coefficients[0]+self.P_coefficients[1]*kair+self.P_coefficients[2]*kair**2+self.P_coefficients[3]*kair**3
        return P

    def calculate_F(self, S):
        kair = self.f / self.f0
        a, b, c = self.H_coefficients[1]/self.number**2-S*self.number**2, self.H_coefficients[2]/self.number*kair, self.H_coefficients[3]*kair**2
        if b**2-4*a*c >= 0:
            Fcw_result = max((-b+(b**2-4*a*c)**0.5)/(2*a), (-b-(b**2-4*a*c)**0.5)/(2*a))
        else:
            Fcw_result = max((-b+(-b**2+4*a*c)**0.5)/(2*a), (-b-(-b**2+4*a*c)**0.5)/(2*a))
        self.Fcw = Fcw_result
        return Fcw_result*self.number


class Cooling_tower:
    
    def __init__(self, Tcwr_path, P_coefficients, f0, Fcw0, number):
        self.Tcwr_path = Tcwr_path
        self.P_coefficients = P_coefficients
        self.f0 = f0
        self.Fcw0 = Fcw0
        self.number = number
        self.f = 40
        self.Fcw = 260
        self.Tcws = 35.5
        self.Twb = 25
    
    def calculate_P(self):
        P = self.P_coefficients[0]+self.P_coefficients[1]*self.f+self.P_coefficients[2]*self.f**2+self.P_coefficients[3]*self.f**3
        return P*self.number
    
    def calculate_Tcwr(self):
        lgb_reg = load(self.Tcwr_path)
        [Tcwr] = lgb_reg.predict(array([[self.Tcws, self.f, self.Twb, self.Fcw]]))
        return Tcwr


if __name__ == "main":
    pass
    
