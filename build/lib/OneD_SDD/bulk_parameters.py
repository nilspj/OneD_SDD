import numpy as np

class BulkParameters:
    def __init__(self,label,type):
        self.label = label
        self.type = type
        pass

class NM_BulkParameters(BulkParameters):
    def __init__(self,De,sigma,l_sf, label = "normal metal"):
        super().__init__(label,"NM")
        self.De = De
        self.sigma = sigma
        self.l_sf = l_sf
        pass

class FM_BulkParameters(BulkParameters):
    def __init__(self,De,sigma,l_sf,l_J,l_phi,m,beta_S,beta_D, label = "ferromagnetic"):
        super().__init__(label,"FM")
        self.De = De
        self.sigma = sigma
        self.l_sf = l_sf
        self.l_J = l_J
        self.l_phi = l_phi
        self.m = m/np.sqrt(np.dot(m,m))
        self.beta_S = beta_S
        self.beta_D = beta_D
        pass

class HM_BulkParameters(BulkParameters):
    def __init__(self,De,sigma,l_sf,theta_SHAy,label = "heavy metal"):
        super().__init__(label,"HM")
        self.De = De
        self.sigma = sigma
        self.l_sf = l_sf
        self.theta_SHAy = theta_SHAy
        pass

class AFM_BulkParameters(BulkParameters):
    def __init__(self,De,sigma,l_sf,theta_SHAx,theta_SHAy,theta_SHAz, label = "anti-ferromagnetic"):
        super().__init__(label,"AFM")
        self.De = De
        self.sigma = sigma
        self.l_sf = l_sf
        self.theta_SHAx = theta_SHAx
        self.theta_SHAy = theta_SHAy
        self.theta_SHAz = theta_SHAz
        pass

class HFM_BulkParameters(BulkParameters):
    def __init__(self,De,sigma,l_sf,l_J,l_phi,m,beta_S,beta_D,theta_AH,xi_AH, theta_AMR,eta_AMR, label = "heavy ferromagnet"):
        super().__init__(label,"HFM")
        self.De = De
        self.sigma = sigma
        self.l_sf = l_sf
        self.l_J = l_J
        self.l_phi = l_phi
        self.m = m
        self.beta_S = beta_S
        self.beta_D = beta_D
        self.theta_AH  = theta_AH
        self.xi_AH = xi_AH
        self.theta_AMR = theta_AMR
        self.eta_AMR = eta_AMR
        pass