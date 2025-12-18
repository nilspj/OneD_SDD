import numpy as np
from .physical_constants import mub_,ee_,ge_,hbar_,me_
from importlib_resources import files

class InterfaceParameters:
    def __init__(self,label,type):
        self.label = label
        self.type = type
        pass

class Continuous_InterfaceParameters(InterfaceParameters):
    def __init__(self,label = "continous interface"):
        super().__init__(label,"continous")
        pass

class RashbaSpinMixing_InterfaceParameters(InterfaceParameters):
    def __init__(self,u0,uEx,uR,uD,kF,t_NM = None ,t_FM = None,mu_approx=True,full_absorption = False,label = "Rashba interface", conductance_switches = None):
        super().__init__(label,"RashbaSpinMixing")
        self.u0 = u0
        self.uEx = uEx
        self.uR = uR
        self.uD = uD
        self.kF = kF
        self.vF = hbar_*kF/me_
        self.t_NM = t_NM
        self.t_FM = t_FM
        self.mu_approx = mu_approx
        self.full_absorption = full_absorption
        self.theta, self.phi, self.weight = self.read_and_sort_quadrature_points()
        self.conductance_switches = conductance_switches
        pass

    
    def read_and_sort_quadrature_points(self):
        print('Reading quadrature points and weights')
        with open(files('OneD_SDD.quadrature_points').joinpath('lebedev_053.txt').__str__(), 'r') as file:
            lines = file.readlines()
        theta = np.array([])
        phi = np.array([])
        weight = np.array([])
        for line in lines:
            values = line.split()
            phi = np.append(phi,float(values[0]))
            theta = np.append(theta,float(values[1]))
            weight = np.append(weight,float(values[2]))
        # remove quadrature points with theta > 90
        ind = np.argwhere(theta > 90.0)
        theta = np.delete(theta,ind)
        phi = np.delete(phi,ind)
        weight = np.delete(weight,ind)
        # sort quadrature points
        ind = np.argsort(theta)
        theta = theta[ind]
        phi = phi[ind]
        weight = weight[ind]
        # convert to radians
        theta = np.pi/180.0*theta
        phi = np.pi/180.0*phi
        return theta,phi,weight
    
class RashbaPerturbSpinMixing_InterfaceParameters(InterfaceParameters):
    def __init__(self,u0,uEx,uR,uD,kF,t_NM = None ,t_FM = None,full_absorption = False,label = "Rashba perturbation interface"):
        super().__init__(label,"RashbaPerturbSpinMixing")
        self.u0 = u0
        self.uEx = uEx
        self.uR = uR
        self.uD = uD
        self.kF = kF
        self.vF = hbar_*kF/me_
        self.t_NM = t_NM
        self.t_FM = t_FM
        self.full_absorption = full_absorption
        pass

class MCT_InterfaceParameters(InterfaceParameters):
    def __init__(self,G_mix,G_up,G_down,label = "MCT interface" ):
        super().__init__(label,"MCT")
        self.G_mix  = G_mix
        self.G_up   = G_up
        self.G_down = G_down
        pass

class MCT_Rashba_InterfaceParameters(InterfaceParameters):
    def __init__(self,G_mix,G_up,G_down,sigma_mix,gamma_mix,sigma_up,sigma_down,T_mix = None ,full_absorption = True,label = "MCT Rashba interface"):
        super().__init__(label,"MCT_Rashba")
        self.G_mix  = G_mix
        self.T_mix = T_mix
        self.G_up   = G_up
        self.G_down = G_down
        self.sigma_mix = sigma_mix
        self.gamma_mix = gamma_mix
        self.sigma_up = sigma_up
        self.sigma_down = sigma_down
        self.full_absorption = full_absorption
        pass

