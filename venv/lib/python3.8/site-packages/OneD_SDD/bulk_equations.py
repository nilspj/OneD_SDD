import numpy as np
from .physical_constants import mub_,ee_

class Layer:
    def __init__(self,electric_field,interface_1,interface_2,material,stack_thickness):
        """
        Initializes a Layer object.

        Args:
            interface_1 (float): The position of the first interface.
            interface_2 (float): The position of the second interface.
            material (str): The material of the layer.
        """
        self.E = electric_field
        self.z_1 = interface_1
        self.z_2 = interface_2
        self.thickness = abs(interface_2 - interface_1)
        self.material = material
        self.stack_thickness = stack_thickness
        pass

    def spin_accumulation(self,A,B,z):
        """
        Calculates the spin accumulation at a given position.

        Args:
            A (float): Parameter A.
            B (float): Parameter B.
            z (float): The position.

        Returns:
            float: The spin accumulation.
        """
        pass

    def spin_current(self,A,B,z,scale):
        """
        Calculates the spin current at a given position.

        Args:
            A (float): Parameter A.
            B (float): Parameter B.
            z (float): The position.

        Returns:
            float: The spin current.
        """
        pass

    def S_low_boundary_M(self):
        """
        Calculates the low boundary of the spin accumulation.

        Returns:
            float: The low boundary of the spin accumulation.
        """
        pass

    def S_high_boundary_M(self):
        """
        Calculates the high boundary of the spin accumulation.

        Returns:
            float: The high boundary of the spin accumulation.
        """
        pass

    def J_low_boundary_M(self): 
        """
        Calculates the low boundary of the spin current.

        Returns:
            float: The low boundary of the spin current.
        """
        pass

    def J_high_boundary_M(self):
        """
        Calculates the high boundary of the spin current.

        Returns:
            float: The high boundary of the spin current.
        """
        pass

    def J_low_boundary_b(self):
        """
        Calculates the low boundary of the spin accumulation independent spin current contribution.

        Returns:
            float: The low boundary  of the spin accumulation independent current contribution.
        """
        pass

    def J_high_boundary_b(self):
        """
        Calculates the high boundary of the spin accumulation independent spin current.

        Returns:
            float: The high boundary of the spin accumulation independent spin current.
        """
        pass   
    
class FM_layer(Layer):
    def __init__(self,electric_field,interface_1,interface_2,bulk_params,stack_thickness):
        super().__init__(electric_field,interface_1,interface_2,'FM',stack_thickness)
        self.m = bulk_params.m
        self.beta_S = bulk_params.beta_S
        self.beta_D = bulk_params.beta_D
        self.De = bulk_params.De
        self.sigma = bulk_params.sigma
        self.l_sf = bulk_params.l_sf
        self.l_J = bulk_params.l_J
        self.l_phi = bulk_params.l_phi
        A = self.build_system_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(A)
        self.l_s = 1.0/np.sqrt(eigenvalues)
        self.v_s = eigenvectors
        pass

    def build_m_outer_m(self):
        M = np.zeros((3,3),dtype = np.complex128)
        M[0,0] = 1.0 - self.beta_S*self.beta_D*self.m[0]*self.m[0]
        M[0,1] =     - self.beta_S*self.beta_D*self.m[0]*self.m[1]
        M[0,2] =     - self.beta_S*self.beta_D*self.m[0]*self.m[2]
        M[1,0] =     - self.beta_S*self.beta_D*self.m[1]*self.m[0]
        M[1,1] = 1.0 - self.beta_S*self.beta_D*self.m[1]*self.m[1]
        M[1,2] =     - self.beta_S*self.beta_D*self.m[1]*self.m[2]
        M[2,0] =     - self.beta_S*self.beta_D*self.m[2]*self.m[0]
        M[2,1] =     - self.beta_S*self.beta_D*self.m[2]*self.m[1]
        M[2,2] = 1.0 - self.beta_S*self.beta_D*self.m[2]*self.m[2]
        return M

    def build_system_matrix(self):
        A = np.zeros((3,3),dtype = np.complex128)
        A[0,0]= 1./self.l_sf**2 + (self.m[1]**2 + self.m[2]**2)/self.l_phi**2
        A[0,1]=  self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[0,2]= -self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[1,0]= -self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[1,1]= 1./self.l_sf**2 + (self.m[0]**2 + self.m[2]**2)/self.l_phi**2
        A[1,2]=  self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,0]=  self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[2,1]= -self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,2]= 1./self.l_sf**2 + (self.m[1]**2 + self.m[0]**2)/self.l_phi**2
        
        MoM = self.build_m_outer_m()
        A = np.linalg.inv(MoM) @ A
        return A
    
    def build_torque_matrix(self):
        A = np.zeros((3,3))
        A[0,0]= (self.m[1]**2 + self.m[2]**2)/self.l_phi**2
        A[0,1]=  self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[0,2]= -self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[1,0]= -self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[1,1]= (self.m[0]**2 + self.m[2]**2)/self.l_phi**2
        A[1,2]=  self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,0]=  self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[2,1]= -self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,2]= (self.m[1]**2 + self.m[0]**2)/self.l_phi**2
        A *= self.sigma
        return A
    
    def charge_potential_and_current(self,A,B,z):
        # Charge potential and current in a normal metal for given coefficients A and B
        Vs = self.spin_accumulation(A,B,z,False)
        Vc =  np.real(A[3]*z + B[3]) + self.beta_D*self.m @ Vs
        Jc =  np.ones(len(z))
        Jc *= -self.sigma*np.real(A[3])
        return np.array([Vc,Jc])

    def spin_accumulation(self,A,B,z,scale):
        v1 = self.v_s[:,0]
        v2 = self.v_s[:,1]
        v3 = self.v_s[:,2]
        Sx  =   np.real(v1[0]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) + B[0]*np.exp(-(z-self.z_1)/self.l_s[0])) 
                      + v2[0]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) + B[1]*np.exp(-(z-self.z_1)/self.l_s[1])) 
                      + v3[0]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) + B[2]*np.exp(-(z-self.z_1)/self.l_s[2])))
        Sy  =   np.real(v1[1]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) + B[0]*np.exp(-(z-self.z_1)/self.l_s[0])) 
                      + v2[1]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) + B[1]*np.exp(-(z-self.z_1)/self.l_s[1])) 
                      + v3[1]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) + B[2]*np.exp(-(z-self.z_1)/self.l_s[2])))
        Sz  =   np.real(v1[2]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) + B[0]*np.exp(-(z-self.z_1)/self.l_s[0])) 
                      + v2[2]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) + B[1]*np.exp(-(z-self.z_1)/self.l_s[1])) 
                      + v3[2]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) + B[2]*np.exp(-(z-self.z_1)/self.l_s[2])))
        S = np.array([Sx,Sy,Sz])
        # convert to units of A/m
        if scale:
            S *= mub_/ee_*self.sigma/self.De
        return S
    
    def spin_torque(self,A,B,z):
        M = self.build_torque_matrix()
        S = self.spin_accumulation(A,B,z,False)
        T = M @ S
        return T
        
    
    def spin_current(self,A,B,z,scale):
        v1 = self.v_s[:,0]
        v2 = self.v_s[:,1]
        v3 = self.v_s[:,2]
        MoM  = np.real(self.build_m_outer_m())
        Jsx  =  -self.sigma*np.real(v1[0]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) - B[0]*np.exp(-(z-self.z_1)/self.l_s[0]))/self.l_s[0] 
                                  + v2[0]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) - B[1]*np.exp(-(z-self.z_1)/self.l_s[1]))/self.l_s[1] 
                                  + v3[0]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) - B[2]*np.exp(-(z-self.z_1)/self.l_s[2]))/self.l_s[2])
        Jsy  =  -self.sigma*np.real(v1[1]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) - B[0]*np.exp(-(z-self.z_1)/self.l_s[0]))/self.l_s[0] 
                                  + v2[1]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) - B[1]*np.exp(-(z-self.z_1)/self.l_s[1]))/self.l_s[1] 
                                  + v3[1]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) - B[2]*np.exp(-(z-self.z_1)/self.l_s[2]))/self.l_s[2])
        Jsz  =  -self.sigma*np.real(v1[2]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) - B[0]*np.exp(-(z-self.z_1)/self.l_s[0]))/self.l_s[0] 
                                  + v2[2]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) - B[1]*np.exp(-(z-self.z_1)/self.l_s[1]))/self.l_s[1] 
                                  + v3[2]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) - B[2]*np.exp(-(z-self.z_1)/self.l_s[2]))/self.l_s[2])
        Js = np.array([Jsx,Jsy,Jsz])
        Js[:] = MoM @ Js[:]
        Js[0,:] +=  self.sigma*self.beta_S*self.m[0]*np.real(A[3])
        Js[1,:] +=  self.sigma*self.beta_S*self.m[1]*np.real(A[3])
        Js[2,:] +=  self.sigma*self.beta_S*self.m[2]*np.real(A[3])
        # convert to units of A/s 
        if scale:
            Js *= mub_/ee_
        return Js
    

    def V_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128)
        M_low[0:3,0] = self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])  # Ax z=z1
        M_low[0:3,1] = self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])  # Ay z=z1
        M_low[0:3,2] = self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])  # Az z=z1
        M_low[3,3]   = self.z_1                                               # Ac z=z1
        M_low[0:3,4] = self.v_s[:,0]                                          # Bx z=z1
        M_low[0:3,5] = self.v_s[:,1]                                          # By z=z1
        M_low[0:3,6] = self.v_s[:,2]                                          # Bz z=z1
        M_low[3,7]   = 1                                                      # Bc z=z1

        # beta_D contribution
        # M_low[3,0]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])
        # M_low[3,1]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])
        # M_low[3,2]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])
        # M_low[3,4]  +=  self.beta_D*self.m @ self.v_s[:,0]
        # M_low[3,5]  +=  self.beta_D*self.m @ self.v_s[:,1]
        # M_low[3,6]  +=  self.beta_D*self.m @ self.v_s[:,2]
        return M_low
    
    def V_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128)
        M_high[0:3,0] = self.v_s[:,0] # Ax z=z2
        M_high[0:3,1] = self.v_s[:,1] # Ay z=z2
        M_high[0:3,2] = self.v_s[:,2] # Az z=z2
        M_high[3,3]   = self.z_2      # Ac z=z2
        M_high[0:3,4] = self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])  # Bx z=z2
        M_high[0:3,5] = self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])  # By z=z2
        M_high[0:3,6] = self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])  # Bz z=z2
        M_high[3,7]   = 1                                                       # Bc z=z2

        # beta_D contribution
        # M_high[3,0]  +=  self.beta_D*self.m @ self.v_s[:,0]
        # M_high[3,1]  +=  self.beta_D*self.m @ self.v_s[:,1]
        # M_high[3,2]  +=  self.beta_D*self.m @ self.v_s[:,2]
        # M_high[3,4]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])
        # M_high[3,5]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])
        # M_high[3,6]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])
        return M_high

    def J_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_low[0:3,0] = -self.sigma*MoM@self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])/self.l_s[0]  # Ax z=z1
        M_low[0:3,1] = -self.sigma*MoM@self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])/self.l_s[1]  # Ay z=z1
        M_low[0:3,2] = -self.sigma*MoM@self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])/self.l_s[2]  # Az z=z1
        M_low[3,3]   = -self.sigma                                                                        # Ac z=z1
        M_low[0:3,4] =  self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Bx z=z1
        M_low[0:3,5] =  self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # By z=z1
        M_low[0:3,6] =  self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Bz z=z1
        M_low[3,7]   =  0                                        # Bc z=z1

        # beta_sigma contribution
        M_low[0:3,3]  +=  self.sigma*self.beta_S*self.m 
        return M_low
    
    def J_low_external_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_low[0:3,0] = -self.sigma*MoM@self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])/self.l_s[0]  # Ax z=z1
        M_low[0:3,1] = -self.sigma*MoM@self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])/self.l_s[1]  # Ay z=z1
        M_low[0:3,2] = -self.sigma*MoM@self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])/self.l_s[2]  # Az z=z1
        M_low[3  ,3] =  self.z_1                                                                          # Ac z=z1
        M_low[0:3,4] =  self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Bx z=z1
        M_low[0:3,5] =  self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # By z=z1
        M_low[0:3,6] =  self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Bz z=z1
        M_low[3  ,7] =  1                                        # Bc z=z1  

        # beta_sigma contribution
        M_low[0:3,3]  +=  self.sigma*self.beta_S*self.m 

        # beta_D contribution
        # M_low[3,0]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])
        # M_low[3,1]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])
        # M_low[3,2]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])
        # M_low[3,4]  +=  self.beta_D*self.m @ self.v_s[:,0]
        # M_low[3,5]  +=  self.beta_D*self.m @ self.v_s[:,1]
        # M_low[3,6]  +=  self.beta_D*self.m @ self.v_s[:,2]
        return M_low

    def J_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_high[0:3,0] = -self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Ax z=z2
        M_high[0:3,1] = -self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # Ay z=z2
        M_high[0:3,2] = -self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Az z=z2
        M_high[  3,3] = -self.sigma                               # Ac z=z2
        M_high[0:3,4] =  self.sigma*MoM@self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])/self.l_s[0] # Bx z=z2
        M_high[0:3,5] =  self.sigma*MoM@self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])/self.l_s[1] # By z=z2
        M_high[0:3,6] =  self.sigma*MoM@self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])/self.l_s[2] # Bz z=z2
        M_high[  3,7] =  0                                                                                 # Bc z=z2

        # beta_sigma contribution
        M_high[0:3,3] += self.sigma*self.beta_S*self.m 
        return M_high
    
    def J_high_external_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_high[0:3,0] = -self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Ax z=z2
        M_high[0:3,1] = -self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # Ay z=z2
        M_high[0:3,2] = -self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Az z=z2
        M_high[  3,3] =  self.z_2                                 # Ac z=z2
        M_high[0:3,4] =  self.sigma*MoM@self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])/self.l_s[0] # Bx z=z2
        M_high[0:3,5] =  self.sigma*MoM@self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])/self.l_s[1] # By z=z2
        M_high[0:3,6] =  self.sigma*MoM@self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])/self.l_s[2] # Bz z=z2
        M_high[  3,7] =  1                                                                                 # Bc z=z2

        # beta_sigma contribution
        M_high[0:3,3] += self.sigma*self.beta_S*self.m 

        # beta_D contribution
        # M_high[3,0]  +=  self.beta_D*self.m @ self.v_s[:,0]
        # M_high[3,1]  +=  self.beta_D*self.m @ self.v_s[:,1]
        # M_high[3,2]  +=  self.beta_D*self.m @ self.v_s[:,2]
        # M_high[3,4]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])
        # M_high[3,5]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])
        # M_high[3,6]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])
        return M_high
    
    def J_low_boundary_b(self):
        return np.array([0.,0.,0.,0.],dtype = np.complex128)
    
    def J_low_external_boundary_b(self):
        return np.array([0.,0.,0,self.E[2]*self.stack_thickness],dtype = np.complex128)

    def J_high_boundary_b(self):
        return np.array([0.,0.,0.,0.],dtype = np.complex128)
    
    def J_high_external_boundary_b(self):
        return np.array([0.,0.,0.,0.],dtype = np.complex128)
        
class NM_layer(Layer):
    def __init__(self,electric_field,interface_1,interface_2,bulk_params,stack_thickness):
        super().__init__(electric_field,interface_1,interface_2,'NM',stack_thickness)
        self.De = bulk_params.De
        self.sigma = bulk_params.sigma
        self.l_sf = bulk_params.l_sf   

    def charge_potential_and_current(self,A,B,z):
        # Charge potential and current in a normal metal for given coefficients A and B
        V  =  np.real(A[3]*z + B[3])
        Jc = np.ones(len(z))
        Jc *= -self.sigma*np.real(A[3])
        return np.array([V,Jc])

    def spin_accumulation(self,A,B,z,scale):   
        # Spin accumulation in a normal metal for given coefficients A and B
        Sx = np.real(A[0]*np.exp((z-self.z_2)/self.l_sf) + B[0]*np.exp(-(z-self.z_1)/self.l_sf))
        Sy = np.real(A[1]*np.exp((z-self.z_2)/self.l_sf) + B[1]*np.exp(-(z-self.z_1)/self.l_sf))
        Sz = np.real(A[2]*np.exp((z-self.z_2)/self.l_sf) + B[2]*np.exp(-(z-self.z_1)/self.l_sf))
        S = np.array([Sx,Sy,Sz])
        # convert to units of A/m
        if scale:
            S *= mub_/ee_*self.sigma/self.De
        return S

    def spin_current(self,A,B,z,scale):
        # Spin current in a normal metal for given coefficients A and B
        Jsx = -self.sigma*np.real(A[0]*np.exp((z-self.z_2)/self.l_sf) - B[0]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf
        Jsy = -self.sigma*np.real(A[1]*np.exp((z-self.z_2)/self.l_sf) - B[1]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf
        Jsz = -self.sigma*np.real(A[2]*np.exp((z-self.z_2)/self.l_sf) - B[2]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf
        Js = np.array([Jsx,Jsy,Jsz])
        # convert to units of A/s 
        if scale:
            Js *= mub_/ee_
        return Js
    

    def V_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Ax z=z1
        M_low[1,1] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Ay z=z1
        M_low[2,2] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Az z=z1
        M_low[3,3] = self.z_1
        M_low[0,4] = 1  # Bx z=z1
        M_low[1,5] = 1  # By z=z1
        M_low[2,6] = 1  # Bz z=z1
        M_low[3,7] = 1
        return M_low
    
    def V_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = 1  # Ax z=z2
        M_high[1,1] = 1  # Ay z=z2
        M_high[2,2] = 1  # Az z=z2
        M_high[3,3] = self.z_2
        M_high[0,4] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # Bx z=z2
        M_high[1,5] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # By z=z2
        M_high[2,6] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # Bz z=z2
        M_high[3,7] = 1
        return M_high  

    def J_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ax z=z1
        M_low[1,1] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ay z=z1
        M_low[2,2] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Az z=z1
        M_low[3,3] = -self.sigma         # Ac z=z1
        M_low[0,4] =  self.sigma/self.l_sf  # Bx z=z1
        M_low[1,5] =  self.sigma/self.l_sf  # By z=z1
        M_low[2,6] =  self.sigma/self.l_sf  # Bz z=z1
        M_low[3,7] =  0                  # Bc z=z1
        return M_low
    
    def J_low_external_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ax z=z1
        M_low[1,1] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ay z=z1
        M_low[2,2] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Az z=z1
        M_low[3,3] =  self.z_1                                                # Ac z=z1
        M_low[0,4] =  self.sigma/self.l_sf  # Bx z=z1
        M_low[1,5] =  self.sigma/self.l_sf  # By z=z1
        M_low[2,6] =  self.sigma/self.l_sf  # Bz z=z1
        M_low[3,7] =  1                     # Bc z=z1
        return M_low

    def J_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = -self.sigma/self.l_sf  # Ax z=z2
        M_high[1,1] = -self.sigma/self.l_sf  # Ay z=z2
        M_high[2,2] = -self.sigma/self.l_sf  # Az z=z2
        M_high[3,3] = -self.sigma         # Ac z=z2
        M_high[0,4] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bx z=z2
        M_high[1,5] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # By z=z2
        M_high[2,6] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bz z=z2
        M_high[3,7] =  0                                                            # Bc z=z2
        return M_high

    def J_high_external_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = -self.sigma/self.l_sf  # Ax z=z2
        M_high[1,1] = -self.sigma/self.l_sf  # Ay z=z2
        M_high[2,2] = -self.sigma/self.l_sf  # Az z=z2
        M_high[3,3] =  self.z_2              # Ac z=z2
        M_high[0,4] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bx z=z2
        M_high[1,5] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # By z=z2
        M_high[2,6] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bz z=z2
        M_high[3,7] =  1                                                            # Bc z=z2
        return M_high      
    
    def J_low_boundary_b(self):
        return np.array([0,0,0,0],dtype = np.complex128)
    
    def J_low_external_boundary_b(self):
        return np.array([0,0,0,self.E[2]*self.stack_thickness],dtype = np.complex128)

    def J_high_boundary_b(self):
        return np.array([0,0,0,0],dtype = np.complex128)
    
    def J_high_external_boundary_b(self):
        return np.array([0,0,0, 0],dtype = np.complex128)
   
class HM_layer(Layer):
    def __init__(self,electric_field,interface_1,interface_2,bulk_params,stack_thickness):
        super().__init__(electric_field,interface_1,interface_2,'HM',stack_thickness)
        self.De         = bulk_params.De
        self.sigma      = bulk_params.sigma
        self.l_sf       = bulk_params.l_sf
        self.theta_SHAy = bulk_params.theta_SHAy
        pass

    def charge_potential_and_current(self,A,B,z):
        # Charge potential and current in a normal metal for given coefficients A and B
        V   =  np.real(A[3]*z + B[3])
        Jc  = np.ones(len(z))
        Jc *= -self.sigma*np.real(A[3])
        return np.array([V,Jc])

    def spin_accumulation(self,A,B,z,scale):
        # Spin accumulation in a normal metal for given coefficients A and B
        Sx = np.real(A[0]*np.exp((z-self.z_2)/self.l_sf) + B[0]*np.exp(-(z-self.z_1)/self.l_sf))
        Sy = np.real(A[1]*np.exp((z-self.z_2)/self.l_sf) + B[1]*np.exp(-(z-self.z_1)/self.l_sf))
        Sz = np.real(A[2]*np.exp((z-self.z_2)/self.l_sf) + B[2]*np.exp(-(z-self.z_1)/self.l_sf))
        S = np.array([Sx,Sy,Sz])
        # convert to units of A/m
        if scale:
            S *= mub_/ee_*self.sigma/self.De
        return S

    def spin_current(self,A,B,z,scale):
        # Spin current in a normal metal for given coefficients A and B
        Jsx = -self.sigma*np.real(A[0]*np.exp((z-self.z_2)/self.l_sf) - B[0]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf
        Jsy = -self.sigma*np.real(A[1]*np.exp((z-self.z_2)/self.l_sf) - B[1]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf - (self.theta_SHAy)*self.sigma*self.E[0]
        Jsz = -self.sigma*np.real(A[2]*np.exp((z-self.z_2)/self.l_sf) - B[2]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf
        Js = np.array([Jsx,Jsy,Jsz])
        # convert to units of A/s 
        if scale:
            Js *= mub_/ee_
        return Js

    def V_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Ax z=z1
        M_low[1,1] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Ay z=z1
        M_low[2,2] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Az z=z1
        M_low[3,3] = self.z_1                               # Ac z=z1
        M_low[0,4] = 1  # Bx z=z1
        M_low[1,5] = 1  # By z=z1
        M_low[2,6] = 1  # Bz z=z1
        M_low[3,7] = 1  # Bc z=z1
        return M_low
    
    def V_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = 1  # Ax z=z2
        M_high[1,1] = 1  # Ay z=z2
        M_high[2,2] = 1  # Az z=z2
        M_high[3,3] = self.z_2  # Ac z=z2
        M_high[0,4] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # Bx z=z2
        M_high[1,5] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # By z=z2
        M_high[2,6] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # Bz z=z2
        M_high[3,7] = 1                                       # Bc z=z2
        return M_high  

    def J_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ax z=z1
        M_low[1,1] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ay z=z1
        M_low[2,2] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Az z=z1
        M_low[3,3] = -self.sigma                                               # Ac z=z1
        M_low[0,4] =  self.sigma/self.l_sf  # Bx z=z1
        M_low[1,5] =  self.sigma/self.l_sf  # By z=z1
        M_low[2,6] =  self.sigma/self.l_sf  # Bz z=z1
        M_low[3,7] =  0                  # Bc z=z1
        return M_low
    
    def J_low_external_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ax z=z1
        M_low[1,1] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ay z=z1
        M_low[2,2] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Az z=z1
        M_low[3,3] =  self.z_1                                               # Ac z=z1
        M_low[0,4] =  self.sigma/self.l_sf  # Bx z=z1
        M_low[1,5] =  self.sigma/self.l_sf  # By z=z1
        M_low[2,6] =  self.sigma/self.l_sf  # Bz z=z1
        M_low[3,7] =  1                     # Bc z=z1
        return M_low

    def J_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = -self.sigma/self.l_sf  # Ax z=z2
        M_high[1,1] = -self.sigma/self.l_sf  # Ay z=z2
        M_high[2,2] = -self.sigma/self.l_sf  # Az z=z2
        M_high[3,3] = -self.sigma         # Ac z=z2
        M_high[0,4] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bx z=z2
        M_high[1,5] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # By z=z2
        M_high[2,6] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bz z=z2
        M_high[3,7] =  0                                                         # Bc z=z2
        return M_high    
    
    def J_high_external_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = -self.sigma/self.l_sf  # Ax z=z2
        M_high[1,1] = -self.sigma/self.l_sf  # Ay z=z2
        M_high[2,2] = -self.sigma/self.l_sf  # Az z=z2
        M_high[3,3] =  self.z_2              # Ac z=z2
        M_high[0,4] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bx z=z2
        M_high[1,5] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # By z=z2
        M_high[2,6] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bz z=z2
        M_high[3,7] =  1                                                         # Bc z=z2
        return M_high   
    
    def J_low_boundary_b(self):
        return np.array([0.,self.theta_SHAy*self.sigma*self.E[0],0.,0],dtype = np.complex128)
    
    def J_low_external_boundary_b(self):
        return np.array([0.,self.theta_SHAy*self.sigma*self.E[0],0., self.E[2]*self.stack_thickness],dtype = np.complex128)

    def J_high_boundary_b(self):
        return np.array([0.,self.theta_SHAy*self.sigma*self.E[0],0.,0],dtype = np.complex128)
    
    def J_high_external_boundary_b(self):
        return np.array([0.,self.theta_SHAy*self.sigma*self.E[0],0., 0],dtype = np.complex128)


class AFM_layer(Layer):
    def __init__(self,electric_field,interface_1,interface_2,bulk_params,stack_thickness):
        super().__init__(electric_field,interface_1,interface_2,'AFM',stack_thickness)
        self.De = bulk_params.De
        self.sigma = bulk_params.sigma
        self.l_sf = bulk_params.l_sf
        self.theta_SHAx = bulk_params.theta_SHAx
        self.theta_SHAy = bulk_params.theta_SHAy
        self.theta_SHAz = bulk_params.theta_SHAz
        pass

    def charge_potential_and_current(self,A,B,z):
        # Charge potential and current in a normal metal for given coefficients A and B
        V  =  np.real(A[3]*z + B[3])
        Jc = np.ones(len(z))
        Jc *= -self.sigma*np.real(A[3])
        return np.array([V,Jc])

    def spin_accumulation(self,A,B,z, scale):
        # Spin accumulation in a normal metal for given coefficients A and B
        Sx = np.real(A[0]*np.exp((z-self.z_2)/self.l_sf) + B[0]*np.exp(-(z-self.z_1)/self.l_sf))
        Sy = np.real(A[1]*np.exp((z-self.z_2)/self.l_sf) + B[1]*np.exp(-(z-self.z_1)/self.l_sf))
        Sz = np.real(A[2]*np.exp((z-self.z_2)/self.l_sf) + B[2]*np.exp(-(z-self.z_1)/self.l_sf))
        S = np.array([Sx,Sy,Sz])
        # convert to units of A/m
        if scale:
            S *= mub_/ee_*self.sigma/self.De
        return S

    def spin_current(self,A,B,z,scale):
        # Spin current in a normal metal for given coefficients A and B
        Jsx = -self.sigma*np.real(A[0]*np.exp((z-self.z_2)/self.l_sf) - B[0]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf + (self.theta_SHAx)*self.sigma*self.E[0]
        Jsy = -self.sigma*np.real(A[1]*np.exp((z-self.z_2)/self.l_sf) - B[1]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf - (self.theta_SHAy)*self.sigma*self.E[0]
        Jsz = -self.sigma*np.real(A[2]*np.exp((z-self.z_2)/self.l_sf) - B[2]*np.exp(-(z-self.z_1)/self.l_sf))/self.l_sf + (self.theta_SHAz)*self.sigma*self.E[0]
        Js = np.array([Jsx,Jsy,Jsz])
        # convert to units of A/s 
        if scale:
            Js *= mub_/ee_
        return Js
    
    def V_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Ax z=z1
        M_low[1,1] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Ay z=z1
        M_low[2,2] = np.exp((self.z_1-self.z_2)/self.l_sf)  # Az z=z1
        M_low[3,3] = self.z_1                               # Ac z=z1
        M_low[0,4] = 1  # Bx z=z1
        M_low[1,5] = 1  # By z=z1
        M_low[2,6] = 1  # Bz z=z1
        M_low[3,7] = 1  # Bc z=z1
        return M_low
    
    def V_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = 1         # Ax z=z2
        M_high[1,1] = 1         # Ay z=z2
        M_high[2,2] = 1         # Az z=z2
        M_high[3,3] = self.z_2  # Ac z=z2
        M_high[0,4] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # Bx z=z2
        M_high[1,5] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # By z=z2
        M_high[2,6] = np.exp(-(self.z_2-self.z_1)/self.l_sf)  # Bz z=z2
        M_high[3,7] = 1                                       # Bc z=z2
        return M_high  

    def J_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ax z=z1
        M_low[1,1] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ay z=z1
        M_low[2,2] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Az z=z1
        M_low[3,3] = -self.sigma                                               # Ac z=z1
        M_low[0,4] =  self.sigma/self.l_sf  # Bx z=z1
        M_low[1,5] =  self.sigma/self.l_sf  # By z=z1
        M_low[2,6] =  self.sigma/self.l_sf  # Bz z=z1
        M_low[3,7] =  0                     # Bc z=z1
        return M_low
    
    def J_low_external_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_low[0,0] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ax z=z1
        M_low[1,1] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Ay z=z1
        M_low[2,2] = -self.sigma*np.exp((self.z_1-self.z_2)/self.l_sf)/self.l_sf  # Az z=z1
        M_low[3,3] =  self.z_1                                                 # Ac z=z1
        M_low[0,4] =  self.sigma/self.l_sf  # Bx z=z1
        M_low[1,5] =  self.sigma/self.l_sf  # By z=z1
        M_low[2,6] =  self.sigma/self.l_sf  # Bz z=z1
        M_low[3,7] =  1                  # Bc z=z1
        return M_low

    def J_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = -self.sigma/self.l_sf  # Ax z=z2
        M_high[1,1] = -self.sigma/self.l_sf  # Ay z=z2
        M_high[2,2] = -self.sigma/self.l_sf  # Az z=z2
        M_high[3,3] = -self.sigma            # Ac z=z2
        M_high[0,4] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bx z=z2
        M_high[1,5] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # By z=z2
        M_high[2,6] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bz z=z2
        M_high[3,7] =  0                                                            # Bc z=z2
        return M_high    
    
    def J_high_external_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128) # spin current matrix
        M_high[0,0] = -self.sigma/self.l_sf  # Ax z=z2
        M_high[1,1] = -self.sigma/self.l_sf  # Ay z=z2
        M_high[2,2] = -self.sigma/self.l_sf  # Az z=z2
        M_high[3,3] =  self.z_2              # Ac z=z2
        M_high[0,4] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bx z=z2
        M_high[1,5] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # By z=z2
        M_high[2,6] =  self.sigma*np.exp(-(self.z_2-self.z_1)/self.l_sf)/self.l_sf  # Bz z=z2
        M_high[3,7] =  1                                                            # Bc z=z2
        return M_high 
    
    def J_low_boundary_b(self):
        return np.array([-self.theta_SHAx*self.sigma*self.E[0],self.theta_SHAy*self.sigma*self.E[0],-self.theta_SHAz*self.sigma*self.E[0],0],dtype = np.complex128)

    def J_low_external_boundary_b(self):
        return np.array([-self.theta_SHAx*self.sigma*self.E[0],self.theta_SHAy*self.sigma*self.E[0],-self.theta_SHAz*self.sigma*self.E[0],          0],dtype = np.complex128)

    def J_high_boundary_b(self):
        return np.array([-self.theta_SHAx*self.sigma*self.E[0],self.theta_SHAy*self.sigma*self.E[0],-self.theta_SHAz*self.sigma*self.E[0], 0],dtype = np.complex128)
    
    def J_high_external_boundary_b(self):
        return np.array([-self.theta_SHAx*self.sigma*self.E[0],self.theta_SHAy*self.sigma*self.E[0],-self.theta_SHAz*self.sigma*self.E[0], -self.E[2]*self.stack_thickness],dtype = np.complex128)
    

class HFM_layer(Layer):
    def __init__(self,electric_field,interface_1,interface_2,bulk_params,stack_thickness):
        super().__init__(electric_field,interface_1,interface_2,'HFM',stack_thickness)
        self.m = bulk_params.m
        self.beta_S = bulk_params.beta_S
        self.beta_D = bulk_params.beta_D
        self.De = bulk_params.De
        self.sigma = bulk_params.sigma
        self.l_sf = bulk_params.l_sf
        self.l_J = bulk_params.l_J
        self.l_phi = bulk_params.l_phi
        self.theta_AH = bulk_params.theta_AH
        self.xi_AH = bulk_params.xi_AH
        self.theta_AMR = bulk_params.theta_AMR
        self.eta_AMR = bulk_params.eta_AMR
        A = self.build_system_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(A)
        self.l_s = 1/np.sqrt(eigenvalues)
        self.v_s = eigenvectors
        pass

    def build_m_outer_m(self):
        M = np.zeros((3,3),dtype = np.complex128)
        M[0,0] = 1 - self.beta_S*self.beta_D*self.m[0]*self.m[0]
        M[0,1] =   - self.beta_S*self.beta_D*self.m[0]*self.m[1]
        M[0,2] =   - self.beta_S*self.beta_D*self.m[0]*self.m[2]
        M[1,0] =   - self.beta_S*self.beta_D*self.m[1]*self.m[0]
        M[1,1] = 1 - self.beta_S*self.beta_D*self.m[1]*self.m[1]
        M[1,2] =   - self.beta_S*self.beta_D*self.m[1]*self.m[2]
        M[2,0] =   - self.beta_S*self.beta_D*self.m[2]*self.m[0]
        M[2,1] =   - self.beta_S*self.beta_D*self.m[2]*self.m[1]
        M[2,2] = 1 - self.beta_S*self.beta_D*self.m[2]*self.m[2]
        return M

    def build_system_matrix(self):
        A = np.zeros((3,3),dtype = np.complex128)
        A[0,0]= 1./self.l_sf**2 + (self.m[1]**2 + self.m[2]**2)/self.l_phi**2
        A[0,1]=  self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[0,2]= -self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[1,0]= -self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[1,1]= 1./self.l_sf**2 + (self.m[0]**2 + self.m[2]**2)/self.l_phi**2
        A[1,2]=  self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,0]=  self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[2,1]= -self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,2]= 1./self.l_sf**2 + (self.m[1]**2 + self.m[0]**2)/self.l_phi**2
        
        MoM = self.build_m_outer_m()
        A = np.linalg.inv(MoM) @ A
        return A
    
    def build_torque_matrix(self):
        A = np.zeros((3,3))
        A[0,0]= (self.m[1]**2 + self.m[2]**2)/self.l_phi**2
        A[0,1]=  self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[0,2]= -self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[1,0]= -self.m[2]/self.l_J**2 - self.m[0]*self.m[1]/self.l_phi**2
        A[1,1]= (self.m[0]**2 + self.m[2]**2)/self.l_phi**2
        A[1,2]=  self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,0]=  self.m[1]/self.l_J**2 - self.m[0]*self.m[2]/self.l_phi**2
        A[2,1]= -self.m[0]/self.l_J**2 - self.m[1]*self.m[2]/self.l_phi**2
        A[2,2]= (self.m[1]**2 + self.m[0]**2)/self.l_phi**2
        A *= self.sigma
        return A
    
    def charge_potential_and_current(self,A,B,z):
        # Charge potential and current in a normal metal for given coefficients A and B
        Vs = self.spin_accumulation(A,B,z,False)
        Vc =  np.real(A[3]*z + B[3]) + self.beta_D*self.m @ Vs
        Jc =  np.ones(len(z))
        Jc *= -self.sigma*np.real(A[3])
        return np.array([Vc,Jc])

    def spin_accumulation(self,A,B,z,scale):
        v1 = self.v_s[:,0]
        v2 = self.v_s[:,1]
        v3 = self.v_s[:,2]
        Sx  =   np.real(v1[0]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) + B[0]*np.exp(-(z-self.z_1)/self.l_s[0])) 
                      + v2[0]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) + B[1]*np.exp(-(z-self.z_1)/self.l_s[1])) 
                      + v3[0]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) + B[2]*np.exp(-(z-self.z_1)/self.l_s[2])))
        Sy  =   np.real(v1[1]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) + B[0]*np.exp(-(z-self.z_1)/self.l_s[0])) 
                      + v2[1]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) + B[1]*np.exp(-(z-self.z_1)/self.l_s[1])) 
                      + v3[1]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) + B[2]*np.exp(-(z-self.z_1)/self.l_s[2])))
        Sz  =   np.real(v1[2]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) + B[0]*np.exp(-(z-self.z_1)/self.l_s[0])) 
                      + v2[2]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) + B[1]*np.exp(-(z-self.z_1)/self.l_s[1])) 
                      + v3[2]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) + B[2]*np.exp(-(z-self.z_1)/self.l_s[2])))
        S = np.array([Sx,Sy,Sz])
        # convert to units of A/m
        if scale:
            S *= mub_/ee_*self.sigma/self.De
        return S
    
    def spin_torque(self,A,B,z):
        M = self.build_torque_matrix()
        S = self.spin_accumulation(A,B,z,False)
        T = M @ S
        return T
        
    
    def spin_current(self,A,B,z,scale):
        v1 = self.v_s[:,0]
        v2 = self.v_s[:,1]
        v3 = self.v_s[:,2]
        MoM  = np.real(self.build_m_outer_m())
        Jsx  =  -self.sigma*np.real(v1[0]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) - B[0]*np.exp(-(z-self.z_1)/self.l_s[0]))/self.l_s[0] 
                                  + v2[0]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) - B[1]*np.exp(-(z-self.z_1)/self.l_s[1]))/self.l_s[1] 
                                  + v3[0]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) - B[2]*np.exp(-(z-self.z_1)/self.l_s[2]))/self.l_s[2])
        Jsy  =  -self.sigma*np.real(v1[1]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) - B[0]*np.exp(-(z-self.z_1)/self.l_s[0]))/self.l_s[0] 
                                  + v2[1]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) - B[1]*np.exp(-(z-self.z_1)/self.l_s[1]))/self.l_s[1] 
                                  + v3[1]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) - B[2]*np.exp(-(z-self.z_1)/self.l_s[2]))/self.l_s[2])
        Jsz  =  -self.sigma*np.real(v1[2]*(A[0]*np.exp((z-self.z_2)/self.l_s[0]) - B[0]*np.exp(-(z-self.z_1)/self.l_s[0]))/self.l_s[0] 
                                  + v2[2]*(A[1]*np.exp((z-self.z_2)/self.l_s[1]) - B[1]*np.exp(-(z-self.z_1)/self.l_s[1]))/self.l_s[1] 
                                  + v3[2]*(A[2]*np.exp((z-self.z_2)/self.l_s[2]) - B[2]*np.exp(-(z-self.z_1)/self.l_s[2]))/self.l_s[2])
        Js = np.array([Jsx,Jsy,Jsz])
        Js[:] = MoM @ Js[:]
        Js[0,:] +=  self.sigma*self.beta_S*self.m[0]*np.real(A[3])
        Js[1,:] +=  self.sigma*self.beta_S*self.m[1]*np.real(A[3])
        Js[2,:] +=  self.sigma*self.beta_S*self.m[2]*np.real(A[3])
        # Anamolous Hall effect and anisotropic magnetoresistance contribution 
        Js[0,:] += self.m[0]*(self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) + self.sigma*self.eta_AMR*self.theta_AMR*self.m[2]*(self.m @ self.E))
        Js[1,:] += self.m[1]*(self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) + self.sigma*self.eta_AMR*self.theta_AMR*self.m[2]*(self.m @ self.E))
        Js[2,:] += self.m[2]*(self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) + self.sigma*self.eta_AMR*self.theta_AMR*self.m[2]*(self.m @ self.E))
        # convert to units of A/s 
        if scale:
            Js *= mub_/ee_
        return Js
    

    def V_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128)
        M_low[0:3,0] = self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])  # Ax z=z1
        M_low[0:3,1] = self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])  # Ay z=z1
        M_low[0:3,2] = self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])  # Az z=z1
        M_low[3,3]   = self.z_1                                               # Ac z=z1
        M_low[0:3,4] = self.v_s[:,0]                                          # Bx z=z1
        M_low[0:3,5] = self.v_s[:,1]                                          # By z=z1
        M_low[0:3,6] = self.v_s[:,2]                                          # Bz z=z1
        M_low[3,7]   = 1                                                      # Bc z=z1

        # beta_D contribution
        M_low[3,0]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])
        M_low[3,1]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])
        M_low[3,2]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])
        M_low[3,4]  +=  self.beta_D*self.m @ self.v_s[:,0]
        M_low[3,5]  +=  self.beta_D*self.m @ self.v_s[:,1]
        M_low[3,6]  +=  self.beta_D*self.m @ self.v_s[:,2]
        return M_low
    
    def V_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128)
        M_high[0:3,0] = self.v_s[:,0] # Ax z=z2
        M_high[0:3,1] = self.v_s[:,1] # Ay z=z2
        M_high[0:3,2] = self.v_s[:,2] # Az z=z2
        M_high[3,3]   = self.z_2      # Ac z=z2
        M_high[0:3,4] = self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])  # Bx z=z2
        M_high[0:3,5] = self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])  # By z=z2
        M_high[0:3,6] = self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])  # Bz z=z2
        M_high[3,7]   = 1                                                       # Bc z=z2

        # beta_D contribution
        M_high[3,0]  +=  self.beta_D*self.m @ self.v_s[:,0]
        M_high[3,1]  +=  self.beta_D*self.m @ self.v_s[:,1]
        M_high[3,2]  +=  self.beta_D*self.m @ self.v_s[:,2]
        M_high[3,4]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])
        M_high[3,5]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])
        M_high[3,6]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])
        return M_high

    def J_low_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_low[0:3,0] = -self.sigma*MoM@self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])/self.l_s[0]  # Ax z=z1
        M_low[0:3,1] = -self.sigma*MoM@self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])/self.l_s[1]  # Ay z=z1
        M_low[0:3,2] = -self.sigma*MoM@self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])/self.l_s[2]  # Az z=z1
        M_low[3,3]   = -self.sigma                                                                    # Ac z=z1
        M_low[0:3,4] =  self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Bx z=z1
        M_low[0:3,5] =  self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # By z=z1
        M_low[0:3,6] =  self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Bz z=z1
        M_low[3,7]   =  0                                    # Bc z=z1

        # beta_sigma contribution
        M_low[0:3,3]  +=  self.sigma*self.beta_S*self.m 
        return M_low
    
    def J_low_external_boundary_M(self):
        M_low = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_low[0:3,0] = -self.sigma*MoM@self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])/self.l_s[0]  # Ax z=z1
        M_low[0:3,1] = -self.sigma*MoM@self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])/self.l_s[1]  # Ay z=z1
        M_low[0:3,2] = -self.sigma*MoM@self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])/self.l_s[2]  # Az z=z1
        M_low[3  ,3] =  self.z_2                                                                      # Ac z=z1
        M_low[0:3,4] =  self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Bx z=z1
        M_low[0:3,5] =  self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # By z=z1
        M_low[0:3,6] =  self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Bz z=z1
        M_low[3  ,7] =  1                                    # Bc z=z1  

        # beta_sigma contribution
        M_low[0:3,3]  +=  self.sigma*self.beta_S*self.m 

        # beta_D contribution
        M_low[3,0:3]  +=  self.beta_D*self.m @ self.v_s[:,0]*np.exp((self.z_1-self.z_2)/self.l_s[0])
        M_low[3,0:3]  +=  self.beta_D*self.m @ self.v_s[:,1]*np.exp((self.z_1-self.z_2)/self.l_s[1])
        M_low[3,0:3]  +=  self.beta_D*self.m @ self.v_s[:,2]*np.exp((self.z_1-self.z_2)/self.l_s[2])
        M_low[3,4:7]  +=  self.beta_D*self.m @ self.v_s[:,0]
        M_low[3,4:7]  +=  self.beta_D*self.m @ self.v_s[:,1]
        M_low[3,4:7]  +=  self.beta_D*self.m @ self.v_s[:,2]
        return M_low

    def J_high_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_high[0:3,0] = -self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Ax z=z2
        M_high[0:3,1] = -self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # Ay z=z2
        M_high[0:3,2] = -self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Az z=z2
        M_high[  3,3] = -self.sigma                               # Ac z=z2
        M_high[0:3,4] =  self.sigma*MoM@self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])/self.l_s[0] # Bx z=z2
        M_high[0:3,5] =  self.sigma*MoM@self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])/self.l_s[1] # By z=z2
        M_high[0:3,6] =  self.sigma*MoM@self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])/self.l_s[2] # Bz z=z2
        M_high[  3,7] =  0                                                                                 # Bc z=z2

        # beta_sigma contribution
        M_high[0:3,3] += self.sigma*self.beta_S*self.m 
        return M_high
    
    def J_high_external_boundary_M(self):
        M_high = np.zeros((4,8),dtype = np.complex128)
        MoM  = self.build_m_outer_m()
        M_high[0:3,0] = -self.sigma*MoM@self.v_s[:,0]/self.l_s[0] # Ax z=z2
        M_high[0:3,1] = -self.sigma*MoM@self.v_s[:,1]/self.l_s[1] # Ay z=z2
        M_high[0:3,2] = -self.sigma*MoM@self.v_s[:,2]/self.l_s[2] # Az z=z2
        M_high[  3,3] =  self.z_2                                 # Ac z=z2
        M_high[0:3,4] =  self.sigma*MoM@self.v_s[:,0]*np.exp(-(self.z_2-self.z_1)/self.l_s[0])/self.l_s[0] # Bx z=z2
        M_high[0:3,5] =  self.sigma*MoM@self.v_s[:,1]*np.exp(-(self.z_2-self.z_1)/self.l_s[1])/self.l_s[1] # By z=z2
        M_high[0:3,6] =  self.sigma*MoM@self.v_s[:,2]*np.exp(-(self.z_2-self.z_1)/self.l_s[2])/self.l_s[2] # Bz z=z2
        M_high[  3,7] =  1                                                                                 # Bc z=z2

        # beta_sigma contribution
        M_high[0:3,3] += self.sigma*self.beta_S*self.m 
        return M_high
    
    def J_low_boundary_b(self):
        b = np.zeros(4,dtype = np.complex128)
        b[:3] = self.m 
        b[:3]*= -self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) - self.eta_AMR*self.theta_AMR*self.m[2]*np.dot(self.m,self.sigma*self.E)
        b[3]  = 0
        return b
    
    def J_low_external_boundary_b(self):
        b = np.zeros(4,dtype = np.complex128)
        b[:3] = self.m 
        b[:3]*= -self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) - self.eta_AMR*self.theta_AMR*self.m[2]*np.dot(self.m,self.sigma*self.E)
        return b

    def J_high_boundary_b(self):
        b = np.zeros(4,dtype = np.complex128)
        b[:3] = self.m 
        b[:3]*= -self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) - self.eta_AMR*self.theta_AMR*self.m[2]*np.dot(self.m,self.sigma*self.E)
        b[3]  = 0
        return b
    
    def J_high_external_boundary_b(self):
        b = np.zeros(4,dtype = np.complex128)
        b[:3] = self.m 
        b[:3]*= -self.xi_AH*self.theta_AH*(self.m[0]*self.sigma*self.E[1] - self.m[1]*self.sigma*self.E[0]) - self.eta_AMR*self.theta_AMR*self.m[2]*np.dot(self.m,self.sigma*self.E)
        b[3]  = -self.E[2]*self.stack_thickness
        return b