import numpy as np
from .physical_constants import ee_, hbar_, me_
from .interface_parameters import RashbaSpinMixing_InterfaceParameters

class RashbaPerturbSpinMixingConductances():
    def __init__(self,beta_S,int_params):
            self.beta_S = beta_S
            self.u0  = int_params.u0
            self.uEx = int_params.uEx
            self.uR  = int_params.uR
            self.uD  = int_params.uD
            self.kF  = int_params.kF
            self.vF  = hbar_*int_params.kF/me_
            self.t_NM = int_params.t_NM
            self.t_FM = int_params.t_FM
            self.t_up   = self.t_FM + self.beta_S*self.t_FM
            self.t_down = self.t_FM - self.beta_S*self.t_FM
            self.u_up = self.u0 - self.uEx
            self.u_down = self.u0 + self.uEx
            # Conductances
            self.G_mix = self.G_up_down(self.u_up, self.u_down)
            self.T_mix = self.T_up_down(self.u_up, self.u_down)
            self.T_mag_mix = self.T_mag_up_down(self.u_up, self.u_down)
            self.G_up = self.G_up_up(self.u_up)
            self.G_down = self.G_down_down(self.u_down)
            # Conductivities
            self.sigma_c = self.sigma_c(self.u_up, self.u_down)
            self.sigma_l = self.sigma_l(self.u_up, self.u_down)
            self.sigma_mix = self.sigma_mix(self.u_up, self.u_down)
            self.gamma_mix = self.gamma_mix(self.u_up, self.u_down)
            pass

    def sigma_c(self,u_up,u_down):
        """
        Returns the interface charge conductivity.
        """
        Im_GRt_up_up_up = self.Im_GRt_up_up_up(u_up)
        Im_GRt_down_down_down = self.Im_GRt_down_down_down(u_down)
        sigma = Im_GRt_up_up_up*self.t_up - Im_GRt_down_down_down*self.t_down - (Im_GRt_up_up_up - Im_GRt_down_down_down)*self.t_NM
        sigma *= self.vF
        return sigma
    
    def sigma_l(self,u_up,u_down):
        """
        Returns the interface charge conductivity.
        """
        Im_GRt_up_up_up = self.Im_GRt_up_up_up(u_up)
        Im_GRt_down_down_down = self.Im_GRt_down_down_down(u_down)
        sigma = Im_GRt_up_up_up*self.t_up + Im_GRt_down_down_down*self.t_down - (Im_GRt_up_up_up + Im_GRt_down_down_down)*self.t_NM
        sigma *= self.vF
        return sigma
    
    def sigma_mix(self,u_up,u_down):
        """
        Returns the interface spin mixing conductivity.
        """
        GRt_down_up_up = self.GRt_down_up_up(u_up,u_down)
        GRt_up_down_down = self.GRt_up_down_down(u_up,u_down)
        sigma = GRt_down_up_up *self.t_up - np.conjugate(GRt_up_down_down)*self.t_down - (GRt_down_up_up - np.conjugate(GRt_up_down_down))*self.t_NM
        sigma *= self.vF
        return sigma
    
    def gamma_mix(self,u_up,u_down):
        """
        Returns the interface spin mixing torkivity.
        """
        TRt_down_up_up = self.TRt_down_up_up(u_up,u_down)
        TRt_up_down_down = self.TRt_up_down_down(u_up,u_down)
        TRr_down_up_up = self.TRr_down_up_up(u_up,u_down)
        TRr_up_down_down = self.TRr_up_down_down(u_up,u_down)
        gamma = TRt_down_up_up *self.t_up - np.conjugate(TRt_up_down_down)*self.t_down - (TRr_down_up_up - np.conjugate(TRr_up_down_down))*self.t_NM
        gamma *= self.vF
        return gamma


    def G_up_down(self,u_up, u_down):
        """
        Returns the spin mixing conductance
        """
        G_up_down  = 1/2 + u_up*u_down/(2*(u_up + u_down))*(u_down*np.log(u_down**2/(1+u_down**2)) + u_up*np.log(u_up**2/(1+u_up**2)))
        G_up_down += 1j*u_up*u_down/(2*(u_up + u_down))*(u_down*(np.pi-2*np.arctan(u_down)) - u_up*(np.pi-2*np.arctan(u_up)))
        G_up_down *= ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return G_up_down
    
    def T_up_down(self,u_up, u_down):
        """
        Returns the spin mixing conductance for transmission.
        """
        T_up_down  = 1/2 - (u_down**3*np.log((1+u_down**2)/(u_down**2))+ u_up**3*np.log((1+u_up**2)/(u_up**2)))/(2*(u_up + u_down))
        T_up_down += 1j* (2*(u_down**2 - u_up**2) -u_down**3*(np.pi-2*np.arctan(u_down)) + u_up**3*(np.pi-2*np.arctan(u_up)))/(2*(u_up + u_down))
        T_up_down  *= ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return T_up_down
    
    def T_mag_up_down(self, u_up, u_down):
        """
        Returns the interface torkance 
        """
        T_mag_up_down  = 1 - (u_down**2*(np.pi-2*np.arctan(u_down)) + u_up**2*(np.pi-2*np.arctan(u_up)))/(2*(u_down + u_up))
        T_mag_up_down += 1j*(u_down**2*np.log((1+u_down**2)/(u_down**2)) + u_up**2*np.log((u_up**2)/(1+u_up**2)))/(2*(u_down + u_up))
        T_mag_up_down *= (u_down-u_up)
        T_mag_up_down *= ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return T_mag_up_down
    
    def G_up_up(self, u_up):
        """ 
        Returns the majority spin conductance.
        """
        G_up_up = 1/2 -u_up**2*np.log((1+u_up**2)/(u_up**2))/2
        G_up_up *= ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return G_up_up
    
    def G_down_down(self, u_down):
         """ 
         Returns the minority spin conductance.
         """
         G_down_down = 1/2 -u_down**2*np.log((1+u_down**2)/(u_down**2))/2
         G_down_down *= ee_**2*self.kF**2/(4*np.pi**2*hbar_)
         return G_down_down
    
    def Im_GRt_up_up_up(self, u_up):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        Im_GR_up_up_up = u_up - (u_up + 2*u_up**3)*np.log((1+u_up**2)/u_up**2)/2
        Im_GR_up_up_up *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return Im_GR_up_up_up
    
    def Im_GRt_down_down_down(self, u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        Im_GRt_down_down_down = u_down - (u_down + 2*u_down**3)*np.log((1+u_down**2)/u_down**2)/2
        Im_GRt_down_down_down *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return Im_GRt_down_down_down
    
    def GRt_down_up_up(self,u_up,u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        GRt_down_up_up = (2*(u_down - u_up)*(u_down + u_up)*(2/3 + u_down**2 + u_up**2) - (u_down**3 + u_down**5)*(np.pi - 2*np.arctan(u_down)) + (u_up**3 + u_up**5)*(np.pi - 2*np.arctan(u_up)))/(2*(u_down - u_up)*(u_down + u_up))
        GRt_down_up_up += 1j * u_down*(((u_down - u_up) * (u_down + u_up)+ (u_down**2 + u_down**4) * np.log(u_down**2 / (1 + u_down**2))+ (u_up**2 + u_up**4) * np.log((1 + u_up**2) / u_up**2))/(2*(u_down - u_up)*(u_down + u_up)))
        GRt_down_up_up *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return GRt_down_up_up
    
    def GRt_up_down_down(self, u_up, u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        GRt_up_down_down = (2*(u_down - u_up)*(u_down + u_up)*(2/3 + u_down**2 + u_up**2) - (u_down**3 + u_down**5)*(np.pi - 2*np.arctan(u_down)) + (u_up**3 + u_up**5)*(np.pi - 2*np.arctan(u_up)))/(2*(u_down - u_up)*(u_down + u_up))
        GRt_up_down_down +=  1j * u_up*(((u_down - u_up) * (u_down + u_up)+ (u_down**2 + u_down**4) * np.log(u_down**2 / (1 + u_down**2))+ (u_up**2 + u_up**4) * np.log((1 + u_up**2) / u_up**2))/(2*(u_down - u_up)*(u_down + u_up)))
        GRt_up_down_down *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return GRt_up_down_down
    
    def TRt_down_up_up(self, u_up, u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        TRt_down_up_up = u_down**2*(2*(u_down - u_up)*(u_down + u_up) - (u_down + u_down**3)*(np.pi - 2*np.arctan(u_down)) + (u_up + u_up**3)*(np.pi - 2*np.arctan(u_up)))/(2*(u_down + u_up)*(u_down - u_up))
        TRt_down_up_up += 1j * u_down*(((u_down - u_up) * (u_down + u_up)+ (u_down**2 + u_down**4) * np.log(u_down**2 / (1 + u_down**2))+ (u_up**2 + u_up**4) * np.log((1 + u_up**2) / u_up**2))/(2*(u_down - u_up)*(u_down + u_up)))
        TRt_down_up_up *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return TRt_down_up_up
    
    def TRt_up_down_down(self, u_up, u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        TRt_up_down_down = u_up**2*(2*(u_down - u_up)*(u_down + u_up) - (u_down + u_down**3)*(np.pi - 2*np.arctan(u_down)) + (u_up + u_up**3)*(np.pi - 2*np.arctan(u_up)))/(2*(u_down + u_up)*(u_down - u_up))
        TRt_up_down_down += 1j * u_up*(((u_down - u_up) * (u_down + u_up)+ (u_down**2 + u_down**4) * np.log(u_down**2 / (1 + u_down**2))+ (u_up**2 + u_up**4) * np.log((1 + u_up**2) / u_up**2))/(2*(u_down - u_up)*(u_down + u_up)))
        TRt_up_down_down *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return TRt_up_down_down
    
    def TRr_down_up_up(self, u_up, u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        TRr_down_up_up  = u_up**2*(2*(u_down - u_up)*(u_down + u_up) - (u_down + u_down**3)*(np.pi-2*np.arctan(u_down)) + (u_up + u_up**3)*(np.pi-2*np.arctan(u_up)))
        TRr_down_up_up += 1j*u_up*((u_down-u_up)*(u_down+u_up) -u_up*(u_down+u_down**3)*np.log((1+u_down**2)/u_down**2) + (u_up**2 -u_down**2*(1+u_up**2)+u_down*(u_up+u_up**3))*np.log((1+u_up**2)/u_up**2))
        TRr_down_up_up += 1j*u_up**5*(np.log((1-u_up*1j)/(u_up*1j)) + np.log(-(1+u_up*1j)/(u_up*1j))) 
        TRr_down_up_up /= (2*(u_down-u_up)*(u_down+u_up))
        TRr_down_up_up *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return TRr_down_up_up
    
    def TRr_up_down_down(self, u_up, u_down):
        """
        Returns the imaginary part of the majority spin conductance.
        """
        TRr_up_down_down  = u_down**2*(2*(u_down-u_up)*(u_down+u_up) - (u_down+u_down**3)*(np.pi-2*np.arctan(u_down)) + (u_up+u_up**3)*(np.pi-2*np.arctan(u_up)))
        TRr_up_down_down += 1j*u_down*((u_down-u_up)*(u_down+u_up) - (1 + u_down**2)*(u_down**2 + u_down*u_up - u_up**2)*np.log((1+u_down**2)/u_down**2) + u_down*(u_up+u_up**3)*np.log((1+u_up**2)/u_up**2))
        TRr_up_down_down /= (2*(u_down-u_up)*(u_down+u_up))
        TRr_up_down_down *= self.uR*ee_**2*self.kF**2/(4*np.pi**2*hbar_)
        return TRr_up_down_down


def main():
    Rashba_int = RashbaSpinMixing_InterfaceParameters(0.42645,0.20055,0.04,0.0,16.0e9,1.12579e-15,6.00881e-15)
    Conductances = RashbaPerturbSpinMixingConductances(beta_S=0.36, int_params=Rashba_int)
    u_up = 0.42645 - 0.20055
    u_down = 0.42645 + 0.20055
    G_up_down = Conductances.G_up_down(u_up, u_down)
    T_up_down = Conductances.T_up_down(u_up, u_down)
    T_mag_up_down = Conductances.T_mag_up_down(u_up, u_down)
    G_up_up = Conductances.G_up_up(u_up)
    G_down_down = Conductances.G_down_down(u_down)
    print("G_up_down:", G_up_down*1e-15)
    print("T_up_down:", T_up_down*1e-15)
    print("T_mag_up_down:", T_mag_up_down*1e-15)
    print("G_up_up:", G_up_up*1e-15)
    print("G_down_down:", G_down_down*1e-15)
    
    print("\n")

    Im_GRt_up_up_up = Conductances.Im_GRt_up_up_up(u_up)
    Im_GRt_down_down_down = Conductances.Im_GRt_down_down_down(u_down)
    GRt_down_up_up = Conductances.GRt_down_up_up(u_up, u_down)
    GRt_up_down_down = Conductances.GRt_up_down_down(u_up, u_down)
    TRt_down_up_up = Conductances.TRt_down_up_up(u_up, u_down)
    TRt_up_down_down = Conductances.TRt_up_down_down(u_up, u_down)
    TRr_down_up_up = Conductances.TRr_down_up_up(u_up, u_down)
    TRr_up_down_down = Conductances.TRr_up_down_down(u_up, u_down)
    print("Im_GRt_up_up:", Im_GRt_up_up_up*1e-15)
    print("Im_GRt_down_down:", Im_GRt_down_down_down*1e-15)
    print("GRt_down_up_up:", GRt_down_up_up*1e-15)
    print("GRt_up_down_down:", GRt_up_down_down*1e-15)
    print("TRt_down_up_up:", TRt_down_up_up*1e-15)
    print("TRt_up_down_down:", TRt_up_down_down*1e-15)
    print("TRr_down_up_up:", TRr_down_up_up*1e-15)
    print("TRr_up_down_down:", TRr_up_down_down*1e-15)

    print("\n")

    sigma_c = Conductances.sigma_c()
    sigma_l = Conductances.sigma_l()
    sigma_mix = Conductances.sigma_mix()
    gamma_mix = Conductances.gamma_mix()
    print("sigma_c:", sigma_c)
    print("sigma_l:", sigma_l)
    print("sigma_mix:", sigma_mix)
    print("gamma_mix:", gamma_mix)

if __name__ == "__main__":
      main()
      