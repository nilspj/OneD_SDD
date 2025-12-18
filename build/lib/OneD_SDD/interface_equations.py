
import numpy as np
from .rashba_BCs import RashbaSpinMixingTensors
from .rashba_perturb_BCs import RashbaPerturbSpinMixingConductances
from .physical_constants import mub_,ee_, me_


class BoundaryCondition:
    def __init__(self,n_layers,position,lower_layer,upper_layer,int_params):
        """
        Initializes an Interface object.

        Args:
            position (float): The position of the interface.
        """
        self.E = lower_layer.E
        self.n_layers = n_layers
        self.position = position
        self.upper_layer = upper_layer
        self.lower_layer = lower_layer
        self.int_params = int_params
        pass
    
    def assemble_M(self):
        """
        Assembles the spin current boundary matrix.
        """
        pass
    
    def assemble_b(self):
        """
        Assembles the boundary condition vector.
        """
        pass

    def u_outer_u(self,u):
        # u outer u matrix
        ux = np.zeros((4,4))
        ux[0,0] = u[0]*u[0]
        ux[0,1] = u[0]*u[1]
        ux[0,2] = u[0]*u[2]
        ux[1,0] = u[1]*u[0]
        ux[1,1] = u[1]*u[1]
        ux[1,2] = u[1]*u[2]
        ux[2,0] = u[2]*u[0]
        ux[2,1] = u[2]*u[1]
        ux[2,2] = u[2]*u[2]
        ux[3,3] = 1.0
        return ux
    
    def u_sc_u_sc(self,u):
        # 0  u.T 
        # u   0   matrix
        ux = np.zeros((4,4))
        ux[3,0] = u[0]
        ux[3,1] = u[1]
        ux[3,2] = u[2]
        ux[0,3] = u[0]
        ux[1,3] = u[1]
        ux[2,3] = u[2]
        return ux
    
    def u_Cross(self,u):
        # [u]_x matrix
        ux = np.zeros((4,4))
        ux[0,1] = -u[2]
        ux[0,2] =  u[1]
        ux[1,0] =  u[2]
        ux[1,2] = -u[0]
        ux[2,0] = -u[1]
        ux[2,1] =  u[0]
        return ux
    
    def u_Cross_u_Cross(self,u):
        # [u]_x[u]_x matrix
        ux = np.zeros((4,4))
        ux[0,0] = -u[1]*u[1] - u[2]*u[2]
        ux[0,1] =  u[0]*u[1]
        ux[0,2] =  u[0]*u[2]
        ux[1,0] =  u[0]*u[1]
        ux[1,1] = -u[0]*u[0] - u[2]*u[2]
        ux[1,2] =  u[1]*u[2]
        ux[2,0] =  u[0]*u[2]
        ux[2,1] =  u[1]*u[2]
        ux[2,2] = -u[0]*u[0] - u[1]*u[1]
        return ux
    
class ContinousBoundaryCondition(BoundaryCondition):
    def __init__(self,n_layers, position,lower_layer,upper_layer,int_params = None):
        super().__init__(n_layers, position,lower_layer,upper_layer,int_params)
        pass   
    
    def assemble_M(self):
        """
        Assembles the spin current boundary matrix.
        """
        sub_M = np.zeros((8,8*self.n_layers),dtype = np.complex128)
        
        sub_M[0:4, (self.position - 1) * 8:self.position * 8] -= self.lower_layer.J_high_boundary_M()
        sub_M[0:4, self.position * 8:(self.position + 1) * 8] += self.upper_layer.J_low_boundary_M()

        conv_lower = np.eye(4,dtype = np.complex128)
        conv_upper = np.eye(4,dtype = np.complex128)
        conv_lower[0:3] *= mub_/ee_*self.lower_layer.sigma/self.lower_layer.De
        conv_upper[0:3] *= mub_/ee_*self.upper_layer.sigma/self.upper_layer.De 
        sub_M[4:8, (self.position - 1) * 8:self.position * 8] -= conv_lower @ self.lower_layer.V_high_boundary_M()
        sub_M[4:8, self.position * 8:(self.position + 1) * 8] += conv_upper @ self.upper_layer.V_low_boundary_M()
        return sub_M
    
    
    def assemble_b(self):
        """
        Assembles the boundary condition vector.
        """
        sub_b = np.zeros(8,dtype = np.complex128)
        sub_b[0:4] -= self.lower_layer.J_high_boundary_b()
        sub_b[0:4] += self.upper_layer.J_low_boundary_b()
        return sub_b

class RashbaSpinMixingBoundaryCondition(BoundaryCondition):
    def __init__(self,n_layers, position,lower_layer,upper_layer,int_params):
        super().__init__(n_layers, position,lower_layer,upper_layer,int_params)
        self.mu_approx = int_params.mu_approx
        self.full_absorption = int_params.full_absorption
        if (self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            if int_params.t_NM is None:
                int_params.t_NM =  (3 * np.pi**2 * me_ * lower_layer.sigma) / (int_params.kF**3 * ee_**2)
            if int_params.t_FM is None:
                int_params.t_FM = (3 * np.pi**2 * me_ * upper_layer.sigma) / (int_params.kF**3 * ee_**2)
            self.beta_S = self.upper_layer.beta_S
            self.m = self.upper_layer.m
            self.spin_tensors = RashbaSpinMixingTensors(self.m,upper_layer.E,self.beta_S ,self.int_params,flip=False)
        else:
            if int_params.t_NM is None:
                int_params.t_NM =  (3 * np.pi**2 * me_ * upper_layer.sigma) / (int_params.kF**3 * ee_**2)
            if int_params.t_FM is None:
                int_params.t_FM = (3 * np.pi**2 * me_ * lower_layer.sigma) / (int_params.kF**3 * ee_**2)
            if self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM':
                self.beta_S = self.lower_layer.beta_S
                self.m = self.lower_layer.m
            else:
                self.beta_S = 0.0
                self.m = np.zeros(3)
            self.spin_tensors = RashbaSpinMixingTensors(self.m,lower_layer.E,self.beta_S,self.int_params,flip=True)
        self.assemble_interface_tensors(int_params.conductance_switches)
        pass
        

    def longitidunal_transformation_tensor(self):
        '''
        Removes transverse components of a vector quantity 
        '''
        A = np.zeros((4,4),dtype = np.complex128)
        A[0,0] = self.m[0]*self.m[0]
        A[0,1] = self.m[0]*self.m[1]
        A[0,2] = self.m[0]*self.m[2]
        A[1,0] = self.m[1]*self.m[0]
        A[1,1] = self.m[1]*self.m[1]
        A[1,2] = self.m[1]*self.m[2]
        A[2,0] = self.m[2]*self.m[0]
        A[2,1] = self.m[2]*self.m[1]
        A[2,2] = self.m[2]*self.m[2]
        A[3,3] = 1
        return A
    
    def transverse_transformation_tensor(self):
        '''
        Removes longitudinal components of a vector quantity 
        '''
        A = np.zeros((3,3))
        A[0,0] = 1 - self.m[0]*self.m[0]
        A[0,1] =   - self.m[0]*self.m[1]
        A[0,2] =   - self.m[0]*self.m[2]
        A[1,0] =   - self.m[1]*self.m[0]
        A[1,1] = 1 - self.m[1]*self.m[1]
        A[1,2] =   - self.m[1]*self.m[2]
        A[2,0] =   - self.m[2]*self.m[0]
        A[2,1] =   - self.m[2]*self.m[1]
        A[2,2] = 1 - self.m[2]*self.m[2]
        return A
    
    def assemble_interface_tensors(self,sw):
        if(self.mu_approx):
            if sw != None:
                self.G_tensor      = self.spin_tensors.conductance_plus*sw[0]  - self.spin_tensors.conductance_minus*sw[1]  - self.spin_tensors.conductance_up_down*sw[2]
                self.T_tensor      = self.spin_tensors.conductance_plus*sw[0]  - self.spin_tensors.conductance_minus*sw[1]  - self.spin_tensors.torkance_up_down*sw[3]
                self.T_tensor_mag  = self.spin_tensors.torkance_mag_plus*sw[4] - self.spin_tensors.torkance_mag_minus*sw[5] - self.spin_tensors.torkance_mag_up_down*sw[6]

                self.sigma_tensor     = self.spin_tensors.conductivity_plus*sw[7]   - self.spin_tensors.conductivity_minus*sw[8]   - self.spin_tensors.torkivity_up_down*sw[9]
                self.gamma_tensor     = self.spin_tensors.conductivity_plus*sw[7]   - self.spin_tensors.conductivity_minus*sw[8]   - self.spin_tensors.conductivity_up_down*sw[10]
                self.gamma_tensor_mag = self.spin_tensors.torkivity_mag_plus*sw[11] - self.spin_tensors.torkivity_mag_minus*sw[12] - self.spin_tensors.torkivity_mag_up_down*sw[13]
            else:
                self.G_tensor = self.spin_tensors.conductance
                self.T_tensor = self.spin_tensors.torkance
                self.T_tensor_mag = self.spin_tensors.torkance_mag
                self.sigma_tensor = self.spin_tensors.conductivity
                self.gamma_tensor = self.spin_tensors.torkivity
                self.gamma_tensor_mag = self.spin_tensors.torkivity_mag

        else:
            self.avg_conductance   = 2*self.spin_tensors.avg_conductance
            self.delta_conductance = 2*self.spin_tensors.delta_conductance
            self.avg_sigma_tensor   = self.spin_tensors.avg_conductivity
            self.delta_sigma_tensor = self.spin_tensors.delta_conductivity

    def assemble_M(self):
        """
        Assembles the spin current boundary matrix.
        """
        sub_M = np.zeros((8,8*self.n_layers),dtype = np.complex128)
        if(self.mu_approx):
            # lower layer
            A = self.longitidunal_transformation_tensor()
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += self.lower_layer.J_high_boundary_M()
            if(self.full_absorption and (self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM')):
                sub_M[0:4, (self.position - 1) * 8:self.position * 8] += -self.G_tensor @ A @ self.lower_layer.V_high_boundary_M()
                sub_M[0:4, self.position * 8:(self.position + 1) * 8] += +self.T_tensor @ A @ self.upper_layer.V_low_boundary_M()
            else:
                sub_M[0:4, (self.position - 1) * 8:self.position * 8] += -self.G_tensor @ self.lower_layer.V_high_boundary_M()
                sub_M[0:4, self.position * 8:(self.position + 1) * 8] += +self.T_tensor @ self.upper_layer.V_low_boundary_M()

            # upper layer
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += self.upper_layer.J_low_boundary_M()
            if(self.full_absorption and (self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM')):
                sub_M[4:8, self.position * 8:(self.position + 1) * 8] += +self.G_tensor @ A @ self.upper_layer.V_low_boundary_M()
                sub_M[4:8, (self.position - 1) * 8:self.position * 8] += -self.T_tensor @ A @ self.lower_layer.V_high_boundary_M()
            else:
                sub_M[4:8, self.position * 8:(self.position + 1) * 8] += +self.G_tensor @ self.upper_layer.V_low_boundary_M()
                sub_M[4:8, (self.position - 1) * 8:self.position * 8] += -self.T_tensor @ self.lower_layer.V_high_boundary_M()

        else:


            # Delta J
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += self.lower_layer.J_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] -= self.upper_layer.J_low_boundary_M()
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] -= self.delta_conductance @ self.lower_layer.V_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] -= self.delta_conductance @ self.upper_layer.V_low_boundary_M()

            # Avg J
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] +=  self.lower_layer.J_high_boundary_M()
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] +=  self.upper_layer.J_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] -=  self.avg_conductance @ self.lower_layer.V_high_boundary_M()
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] -= -self.avg_conductance @ self.upper_layer.V_low_boundary_M()

        return sub_M
    
    def assemble_b(self):
        """
        Assembles the boundary condition vector.
        """
        sub_b = np.zeros(8,dtype = np.complex128)
        
        if(self.mu_approx):
            A = self.longitidunal_transformation_tensor()
            sub_b[0:4] += self.lower_layer.J_high_boundary_b()
            sub_b[4:8] += self.upper_layer.J_low_boundary_b()
            if(self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
                sub_b[0:4] += self.sigma_tensor*abs(self.E[0])
                if(self.full_absorption):
                    sub_b[4:8] += A @ self.gamma_tensor*abs(self.E[0])
                else:
                    sub_b[4:8] += self.gamma_tensor*abs(self.E[0])
            else:
                if(self.full_absorption):
                    sub_b[0:4] += A @ self.gamma_tensor*abs(self.E[0])
                else:
                    sub_b[0:4] += self.gamma_tensor*abs(self.E[0])
                sub_b[4:8] += self.sigma_tensor*abs(self.E[0])

        else:
            sub_b[0:4] += self.lower_layer.J_high_boundary_b() - self.upper_layer.J_low_boundary_b()
            sub_b[4:8] += self.lower_layer.J_high_boundary_b() + self.upper_layer.J_low_boundary_b()
            if(self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
                sub_b[0:4] += self.delta_sigma_tensor*abs(self.E[0])
                sub_b[4:8] += self.avg_sigma_tensor*abs(self.E[0])
            else:  
                sub_b[0:4] += self.delta_sigma_tensor*abs(self.E[0])
                sub_b[4:8] += -self.avg_sigma_tensor*abs(self.E[0])
        return sub_b
    

class RashbaPerturbSpinMixingBoundaryCondition(BoundaryCondition):
    def __init__(self,n_layers, position,lower_layer,upper_layer,int_params):
        super().__init__(n_layers, position,lower_layer,upper_layer,int_params)

        self.full_absorption = int_params.full_absorption
        if (self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            if int_params.t_NM is None:
                int_params.t_NM =  (3 * np.pi**2 * me_ * lower_layer.sigma) / (int_params.kF**3 * ee_**2)
            if int_params.t_FM is None:
                int_params.t_FM = (3 * np.pi**2 * me_ * upper_layer.sigma) / (int_params.kF**3 * ee_**2)
            self.beta_S = self.upper_layer.beta_S
            self.m = self.upper_layer.m
        else:
            if int_params.t_NM is None:
                int_params.t_NM =  (3 * np.pi**2 * me_ * upper_layer.sigma) / (int_params.kF**3 * ee_**2)
            if int_params.t_FM is None:
                int_params.t_FM = (3 * np.pi**2 * me_ * lower_layer.sigma) / (int_params.kF**3 * ee_**2)
            if self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM':
                self.beta_S = self.lower_layer.beta_S
                self.m = self.lower_layer.m
            else:
                self.beta_S = 0.0
                self.m = np.zeros(3)
        self.spin_tensors = RashbaPerturbSpinMixingConductances(self.beta_S ,self.int_params)
        # Conductances
        self.G_mix = self.spin_tensors.G_mix
        self.T_mix = self.spin_tensors.T_mix
        self.T_mag_mix = self.spin_tensors.T_mag_mix
        # print(self.G_mix*1e-14)
        # print(self.T_mix*1e-14)
        self.G_u   = self.spin_tensors.G_up
        self.G_d   = self.spin_tensors.G_down
        # Conductivities
        self.sigma_mix = self.spin_tensors.sigma_mix
        self.gamma_mix = self.spin_tensors.gamma_mix
        self.sigma_c   = self.spin_tensors.sigma_c
        self.sigma_l   = self.spin_tensors.sigma_l
        # print(self.sigma_mix*1e-6)
        # print(self.gamma_mix*1e-6)
        # print(self.sigma_c*1e-6)
        # print(self.sigma_l*1e-6)
        pass
    
    def longitidunal_transformation_tensor(self):
        '''
        Removes transverse components of a vector quantity 
        '''
        A = np.zeros((4,4),dtype = np.complex128)
        A[0,0] = self.m[0]*self.m[0]
        A[0,1] = self.m[0]*self.m[1]
        A[0,2] = self.m[0]*self.m[2]
        A[1,0] = self.m[1]*self.m[0]
        A[1,1] = self.m[1]*self.m[1]
        A[1,2] = self.m[1]*self.m[2]
        A[2,0] = self.m[2]*self.m[0]
        A[2,1] = self.m[2]*self.m[1]
        A[2,2] = self.m[2]*self.m[2]
        A[3,3] = 1
        return A

    def MCT_transverse_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.real(self.G_mix)*self.u_Cross_u_Cross(self.m)
        A +=  2*np.imag(self.G_mix)*self.u_Cross(self.m) 
        return A
    
    def MCT_transverse_transmission_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.real(self.T_mix)*self.u_Cross_u_Cross(self.m)
        A += -2*np.imag(self.T_mix)*self.u_Cross(self.m) 
        return A
    
    def MCT_longitudinal_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        Gp = (self.G_u + self.G_d)
        Gm = (self.G_u - self.G_d)
        A  = self.u_outer_u(self.m) * Gp
        A -= self.u_sc_u_sc(self.m) * Gm
        return A        
    
    def conductivity_tensor(self):
        sigma = np.zeros(4)
        Exz = np.zeros(3)
        Exz[0] = self.E[1]
        Exz[1] =-self.E[0]
        m_outer_m = self.u_outer_u(self.m)[:3,:3]
        m_cross = self.u_Cross(self.m)[:3,:3]
        m_cross_m_cross = self.u_Cross_u_Cross(self.m)[:3,:3]
        sigma[:3]  =  np.real(self.sigma_mix) * m_cross @ Exz 
        sigma[:3] +=  np.imag(self.sigma_mix) * m_cross_m_cross @ Exz 
        sigma[:3] += -self.sigma_l * m_outer_m @ Exz 
        sigma[3]  +=  self.sigma_c * self.m @ Exz 
        return sigma
    
    def torkivity_tensor(self):
        sigma = np.zeros(4)
        Exz = np.zeros(3)
        Exz[0] = self.E[1]
        Exz[1] =-self.E[0]
        m_outer_m = self.u_outer_u(self.m)[:3,:3]
        m_cross = self.u_Cross(self.m)[:3,:3]
        m_cross_m_cross = self.u_Cross_u_Cross(self.m)[:3,:3]
        sigma[:3]   =  np.real(self.gamma_mix) * m_cross @ Exz 
        sigma[:3]  +=  np.imag(self.gamma_mix) * m_cross_m_cross @ Exz 
        sigma[:3]  += -self.sigma_l * m_outer_m @ Exz 
        sigma[3]   +=  self.sigma_c * self.m @ Exz 
        return sigma
    
    def torkance_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.imag(self.T_mag_mix)*self.u_Cross_u_Cross(self.m)
        A +=  2*np.real(self.T_mag_mix)*self.u_Cross(self.m) 
        return A
    
    def assemble_M(self):
        """
        Assembles the spin current boundary matrix.
        """
        sub_M = np.zeros((8,8*self.n_layers),dtype = np.complex128)
        G_perp_tensor = self.MCT_transverse_tensor()
        T_perp_tensor = self.MCT_transverse_transmission_tensor()
        G_par_tensor  = self.MCT_longitudinal_tensor()
        self.G_tensor = G_par_tensor + G_perp_tensor 
        self.T_tensor = G_par_tensor + G_perp_tensor
        # lower layer
        A = self.longitidunal_transformation_tensor()
        sub_M[0:4, (self.position - 1) * 8:self.position * 8] += self.lower_layer.J_high_boundary_M()
        if(self.full_absorption and (self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM')):
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += -self.G_tensor @ A @ self.lower_layer.V_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] += +self.T_tensor @ A @ self.upper_layer.V_low_boundary_M()
        else:
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += -self.G_tensor @ self.lower_layer.V_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] += +self.T_tensor @ self.upper_layer.V_low_boundary_M()

        # upper layer
        sub_M[4:8, self.position * 8:(self.position + 1) * 8] += self.upper_layer.J_low_boundary_M()
        if(self.full_absorption and (self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM')):
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += +self.G_tensor @ A @ self.upper_layer.V_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] += -self.T_tensor @ A @ self.lower_layer.V_high_boundary_M()
        else:
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += +self.G_tensor @ self.upper_layer.V_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] += -self.T_tensor @ self.lower_layer.V_high_boundary_M()
        return sub_M
    
    
    def assemble_b(self):
        """
        Assembles the boundary condition vector.
        """
        sub_b = np.zeros(8,dtype = np.complex128)
        A = self.longitidunal_transformation_tensor()
        sub_b[0:4] += self.lower_layer.J_high_boundary_b()
        sub_b[4:8] += self.upper_layer.J_low_boundary_b()
        sigmaE = self.conductivity_tensor()
        gammaE = self.torkivity_tensor()
        if(self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            sub_b[0:4] += sigmaE
            if(self.full_absorption):
                sub_b[4:8] += A @ gammaE
            else:
                sub_b[4:8] += gammaE
        else:
            if(self.full_absorption):
                sub_b[0:4] += A @ gammaE
            else:
                sub_b[0:4] += gammaE
            sub_b[4:8] += sigmaE
        return sub_b
    

class Magnetoelectronic_circuit_theory(BoundaryCondition):
    def __init__(self,n_layers,position,lower_layer,upper_layer, int_params):
        super().__init__(n_layers, position,lower_layer,upper_layer,int_params)
        self.G_mix = int_params.G_mix
        self.G_u   = int_params.G_up
        self.G_d   = int_params.G_down
        if(self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            self.m = self.upper_layer.m
        elif (self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM'):
            self.m = self.lower_layer.m
        else:
            self.m = np.zeros(3)

    def MCT_transverse_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.real(self.G_mix)*self.u_Cross_u_Cross(self.m)
        A +=  2*np.imag(self.G_mix)*self.u_Cross(self.m) 
        return A
    
    def MCT_longitudinal_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        Gp = (self.G_u + self.G_d)
        Gm = (self.G_u - self.G_d)
        A[0,0] = self.m[0]*self.m[0]*Gp
        A[0,1] = self.m[0]*self.m[1]*Gp
        A[0,2] = self.m[0]*self.m[2]*Gp
        A[1,0] = self.m[1]*self.m[0]*Gp
        A[1,1] = self.m[1]*self.m[1]*Gp
        A[1,2] = self.m[1]*self.m[2]*Gp
        A[2,0] = self.m[2]*self.m[0]*Gp
        A[2,1] = self.m[2]*self.m[1]*Gp
        A[2,2] = self.m[2]*self.m[2]*Gp
        A[3,0] = -self.m[0]*Gm 
        A[3,1] = -self.m[1]*Gm
        A[3,2] = -self.m[2]*Gm 
        A[0,3] = -self.m[0]*Gm 
        A[1,3] = -self.m[1]*Gm
        A[2,3] = -self.m[2]*Gm
        A[3,3] = Gp
        return A        
    
    def assemble_M(self):
        """
        Assembles the spin current boundary matrix.
        """
        sub_M = np.zeros((8,8*self.n_layers),dtype = np.complex128)
        if(self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            G_perp_tensor = self.MCT_transverse_tensor()
            G_par_tensor  = self.MCT_longitudinal_tensor()
            # NM
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += self.lower_layer.J_high_boundary_M()
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += - G_par_tensor @ self.lower_layer.V_high_boundary_M() - G_perp_tensor @ self.lower_layer.V_high_boundary_M() 
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] += + G_par_tensor @ self.upper_layer.V_low_boundary_M()
            # FM
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += self.upper_layer.J_low_boundary_M()
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += + G_par_tensor @ self.upper_layer.V_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] += - G_par_tensor @ self.lower_layer.V_high_boundary_M()
        elif(self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM'):
            G_perp_tensor = self.MCT_transverse_tensor()
            G_par_tensor  = self.MCT_longitudinal_tensor()
            # FM
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += self.lower_layer.J_high_boundary_M()
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += - G_par_tensor @ self.lower_layer.V_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] += + G_par_tensor @ self.upper_layer.V_low_boundary_M()
            # NM
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += self.upper_layer.J_low_boundary_M()
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += + G_par_tensor @ self.upper_layer.V_low_boundary_M() + G_perp_tensor @ self.upper_layer.V_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] += - G_par_tensor @ self.lower_layer.V_high_boundary_M()
        return sub_M
    
    
    def assemble_b(self):
        sub_b = np.zeros(8,dtype = np.complex128)
        sub_b[0:4] += self.lower_layer.J_high_boundary_b()
        sub_b[4:8] += self.upper_layer.J_low_boundary_b()
        return sub_b
    

class Magnetoelectronic_circuit_theory_with_Rashba(BoundaryCondition):

    def __init__(self,n_layers, position,lower_layer,upper_layer,int_params):
        super().__init__(n_layers, position,lower_layer,upper_layer,int_params)

        self.full_absorption = int_params.full_absorption
        if (self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            self.m = self.upper_layer.m
        else:
            if self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM':
                self.m = self.lower_layer.m
            else:
                self.m = np.zeros(3)
        # Conductances
        self.G_mix = self.int_params.G_mix
        if(self.int_params.T_mix == None):
            self.transmission = False
        else:
            self.transmission = True
            self.T_mix = self.int_params.T_mix
        self.G_u   = self.int_params.G_up
        self.G_d   = self.int_params.G_down
        # Conductivities
        self.sigma_mix = self.int_params.sigma_mix
        self.gamma_mix = self.int_params.gamma_mix
        self.sigma_up   = self.int_params.sigma_up
        self.sigma_down   = self.int_params.sigma_down
        pass
            
    
    def longitidunal_transformation_tensor(self):
        '''
        Removes transverse components of a vector quantity 
        '''
        A = np.zeros((4,4),dtype = np.complex128)
        A[0,0] = self.m[0]*self.m[0]
        A[0,1] = self.m[0]*self.m[1]
        A[0,2] = self.m[0]*self.m[2]
        A[1,0] = self.m[1]*self.m[0]
        A[1,1] = self.m[1]*self.m[1]
        A[1,2] = self.m[1]*self.m[2]
        A[2,0] = self.m[2]*self.m[0]
        A[2,1] = self.m[2]*self.m[1]
        A[2,2] = self.m[2]*self.m[2]
        A[3,3] = 1
        return A

    def MCT_transverse_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.real(self.G_mix)*self.u_Cross_u_Cross(self.m)
        A +=  2*np.imag(self.G_mix)*self.u_Cross(self.m) 
        return A
    
    def MCT_transverse_transmission_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.real(self.T_mix)*self.u_Cross_u_Cross(self.m)
        A += -2*np.imag(self.T_mix)*self.u_Cross(self.m) 
        return A
    
    def MCT_longitudinal_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        Gp = (self.G_u + self.G_d)
        Gm = (self.G_u - self.G_d)
        A  = self.u_outer_u(self.m) * Gp
        A -= self.u_sc_u_sc(self.m) * Gm
        return A       

    def conductivity_tensor(self):
        sigma = np.zeros(4)
        Exz = np.zeros(3)
        Exz[0] = self.E[1]
        Exz[1] =-self.E[0]
        m_outer_m = self.u_outer_u(self.m)[:3,:3]
        m_cross = self.u_Cross(self.m)[:3,:3]
        m_cross_m_cross = self.u_Cross_u_Cross(self.m)[:3,:3]
        sigma[:3]  =  np.real(self.sigma_mix) * m_cross @ Exz 
        sigma[:3] +=  np.imag(self.sigma_mix) * m_cross_m_cross @ Exz 
        sigma[:3] +=  (self.sigma_up + self.sigma_down) * m_outer_m @ Exz 
        sigma[3]  +=  (self.sigma_up - self.sigma_down) * self.m @ Exz 
        return sigma
    
    def torkivity_tensor(self):
        gamma = np.zeros(4)
        Exz = np.zeros(3)
        Exz[0] = self.E[1]
        Exz[1] =-self.E[0]
        m_outer_m = self.u_outer_u(self.m)[:3,:3]
        m_cross = self.u_Cross(self.m)[:3,:3]
        m_cross_m_cross = self.u_Cross_u_Cross(self.m)[:3,:3]
        gamma[:3]  =  np.real(self.gamma_mix) * m_cross @ Exz 
        gamma[:3] +=  np.imag(self.sigma_mix) * m_cross_m_cross @ Exz 
        gamma[:3] +=  (self.sigma_up + self.sigma_down) * m_outer_m @ Exz 
        gamma[3]  +=  (self.sigma_up - self.sigma_down) * self.m @ Exz 
        return gamma    
    
    def torkance_tensor(self):
        A = np.zeros((4,4),dtype = np.complex128)
        A  = -2*np.real(self.G_mix - self.T_mix)*self.u_Cross_u_Cross(self.m)
        A +=  2*np.imag(self.G_mix + self.T_mix)*self.u_Cross(self.m) 
        return A
    
    def assemble_M(self):
        """
        Assembles the spin current boundary matrix.
        """
        sub_M = np.zeros((8,8*self.n_layers),dtype = np.complex128)
        G_perp_tensor = self.MCT_transverse_tensor()
        G_par_tensor  = self.MCT_longitudinal_tensor()
        self.G_tensor = G_par_tensor + G_perp_tensor 
        if (self.transmission):
            T_perp_tensor = self.MCT_transverse_transmission_tensor()
            self.T_tensor = G_par_tensor + T_perp_tensor
        else:
            self.T_tensor = G_par_tensor + G_perp_tensor
        # lower layer
        A = self.longitidunal_transformation_tensor()
        sub_M[0:4, (self.position - 1) * 8:self.position * 8] += self.lower_layer.J_high_boundary_M()
        if(self.full_absorption and (self.lower_layer.material == 'FM' or self.lower_layer.material == 'HFM')):
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += -self.G_tensor @ A @ self.lower_layer.V_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] += +self.T_tensor @ A @ self.upper_layer.V_low_boundary_M()
        else:
            sub_M[0:4, (self.position - 1) * 8:self.position * 8] += -self.G_tensor @ self.lower_layer.V_high_boundary_M()
            sub_M[0:4, self.position * 8:(self.position + 1) * 8] += +self.T_tensor @ self.upper_layer.V_low_boundary_M()

        # upper layer
        sub_M[4:8, self.position * 8:(self.position + 1) * 8] += self.upper_layer.J_low_boundary_M()
        if(self.full_absorption and (self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM')):
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += +self.G_tensor @ A @ self.upper_layer.V_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] += -self.T_tensor @ A @ self.lower_layer.V_high_boundary_M()
        else:
            sub_M[4:8, self.position * 8:(self.position + 1) * 8] += +self.G_tensor @ self.upper_layer.V_low_boundary_M()
            sub_M[4:8, (self.position - 1) * 8:self.position * 8] += -self.T_tensor @ self.lower_layer.V_high_boundary_M()
        return sub_M
    
    
    def assemble_b(self):
        """
        Assembles the boundary condition vector.
        """
        sub_b = np.zeros(8,dtype = np.complex128)
        A = self.longitidunal_transformation_tensor()
        sub_b[0:4] += self.lower_layer.J_high_boundary_b()
        sub_b[4:8] += self.upper_layer.J_low_boundary_b()
        sigmaE = self.conductivity_tensor()
        gammaE = self.torkivity_tensor()
        if(self.upper_layer.material == 'FM' or self.upper_layer.material == 'HFM'):
            sub_b[0:4] += sigmaE
            if(self.full_absorption):
                sub_b[4:8] += A @ gammaE
            else:
                sub_b[4:8] += gammaE
        else:
            if(self.full_absorption):
                sub_b[0:4] += A @ gammaE
            else:
                sub_b[0:4] += gammaE
            sub_b[4:8] += sigmaE
        return sub_b