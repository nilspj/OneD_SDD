import time
import numpy as np
from .physical_constants import mub_,ee_, hbar_, me_
from .interface_parameters import RashbaSpinMixing_InterfaceParameters


class RashbaSpinMixingTensors():
    
    def __init__(self,m,E,beta_S,int_params,flip=False):
        OrtTransf = self.Lab_frame_orthogonal_transformation(E,flip)
        self.m = OrtTransf[:3,:3] @ m
        self.beta_S = beta_S
        self.u0  = int_params.u0
        self.uEx = int_params.uEx
        self.uR  = int_params.uR
        self.uD  = int_params.uD
        self.kF  = int_params.kF
        self.vF  = hbar_*int_params.kF/me_
        self.t_NM = int_params.t_NM
        self.t_FM = int_params.t_FM
        self.mu_approx = int_params.mu_approx
        self.theta, self.phi, self.weight = int_params.theta, int_params.phi, int_params.weight
        # Compute conductance matricies
        (G_0_plus, G_0_minus, Re_G_0_up_down, Im_G_0_up_down, Re_T_0_up_down, Im_T_0_up_down, U_0_Plus, U_0_minus, Re_U_0_up_down, Im_U_0_up_down, 
            G_1_plus, G_1_minus, Re_G_1_up_down, Im_G_1_up_down, Re_T_1_up_down, Im_T_1_up_down, U_1_Plus, U_1_minus, Re_U_1_up_down, Im_U_1_up_down) = self.Compute_scattering_matricies()
        
        # shifted distributions
        # NM
        nu_N = np.zeros(4)
        nu_N[3] = 1.0
        nu_N *= self.vF * self.t_NM

        # FM
        nu_F = np.zeros(4)
        nu_F[0] = -self.beta_S * self.m[0]
        nu_F[1] = -self.beta_S * self.m[1]
        nu_F[2] = -self.beta_S * self.m[2]
        nu_F[3] = 1.0
        nu_F *= self.vF * self.t_FM

        # conductance tensor components
        conductance_plus     =   G_0_plus
        conductance_minus    = - G_0_minus
        conductance_up_down  = - 2*Re_G_0_up_down - 2*Im_G_0_up_down
        torkance_up_down     = - 2*Re_T_0_up_down - 2*Im_T_0_up_down
        torkance_mag_plus    =  2*self.uEx*self.mcross() @ (U_0_Plus)
        torkance_mag_minus   =  2*self.uEx*self.mcross() @ (- U_0_minus)
        torkance_mag_up_down =  2*self.uEx*self.mcross() @ (- 2*Re_U_0_up_down - 2*Im_U_0_up_down)

        # conductivity tensor components
        conductivity_plus    = (G_1_plus)  @ (nu_N-nu_F)
        conductivity_minus   = (- G_1_minus) @ (nu_N-nu_F)
        conductivity_up_down = (- 2*Re_G_1_up_down - 2*Im_G_1_up_down) @ (nu_F)
        torkivity_up_down    = (- 2*Re_T_1_up_down - 2*Im_T_1_up_down) @ (nu_F)
        torkivity_mag_plus   = 2*self.uEx*(self.mcross() @ (U_1_Plus))  @ (nu_F+nu_N)
        torkivity_mag_minus  = 2*self.uEx*(self.mcross() @ (- U_1_minus)) @ (nu_F+nu_N)
        torkivity_mag_up_down= 2*self.uEx*(self.mcross() @ (- 2*Re_U_1_up_down - 2*Im_U_1_up_down)) @ (nu_F+nu_N)

        # Spin accumulation components
        D_plus    = -(U_0_Plus)/self.vF
        D_minus   = -(U_0_minus)/self.vF
        D_up_down = -(2*Re_U_0_up_down - 2*Im_U_0_up_down)/self.vF
        c_plus    = -(U_1_Plus) @ (nu_F+nu_N)/self.vF
        c_minus   = -(U_1_minus) @ (nu_F+nu_N)/self.vF
        c_up_down = -(2*Re_U_1_up_down - 2*Im_U_1_up_down) @ (nu_F)/self.vF

        self.conductance_plus      = OrtTransf.T @ conductance_plus     @ OrtTransf
        self.conductance_minus     = OrtTransf.T @ conductance_minus    @ OrtTransf
        self.conductance_up_down   = OrtTransf.T @ conductance_up_down  @ OrtTransf
        self.torkance_up_down      = OrtTransf.T @ torkance_up_down     @ OrtTransf
        self.torkance_mag_plus     = OrtTransf.T @ torkance_mag_plus    @ OrtTransf
        self.torkance_mag_minus    = OrtTransf.T @ torkance_mag_minus   @ OrtTransf
        self.torkance_mag_up_down  = OrtTransf.T @ torkance_mag_up_down @ OrtTransf

        self.conductivity_plus     = OrtTransf.T @ conductivity_plus
        self.conductivity_minus    = OrtTransf.T @ conductivity_minus
        self.conductivity_up_down  = OrtTransf.T @ conductivity_up_down
        self.torkivity_up_down     = OrtTransf.T @ torkivity_up_down
        self.torkivity_mag_plus    = OrtTransf.T @ torkivity_mag_plus
        self.torkivity_mag_minus   = OrtTransf.T @ torkivity_mag_minus
        self.torkivity_mag_up_down = OrtTransf.T @ torkivity_mag_up_down

        self.D_plus    = OrtTransf.T @ D_plus @ OrtTransf
        self.D_minus   = OrtTransf.T @ D_minus @ OrtTransf
        self.D_up_down = OrtTransf.T @ D_up_down @ OrtTransf
        self.c_plus    = OrtTransf.T @ c_plus
        self.c_minus   = OrtTransf.T @ c_minus
        self.c_up_down = OrtTransf.T @ c_up_down
        ##############################################

        conductance_tensor   = G_0_plus - G_0_minus - 2*Re_G_0_up_down - 2*Im_G_0_up_down
        torkance_tensor      = G_0_plus - G_0_minus - 2*Re_T_0_up_down - 2*Im_T_0_up_down
        torkance_tensor_mag  = 2*self.uEx*self.mcross()@(U_0_Plus - U_0_minus - 2*Re_U_0_up_down - 2*Im_U_0_up_down)

        conductivity_tensor  = G_1_plus - G_1_minus - 2*Re_G_1_up_down - 2*Im_G_1_up_down
        torkivity_tensor     = G_1_plus - G_1_minus - 2*Re_T_1_up_down - 2*Im_T_1_up_down
        torkivity_tensor_mag = 2*self.uEx*self.mcross()@(U_1_Plus - U_1_minus - 2*Re_U_1_up_down - 2*Im_U_1_up_down)

        conductivity       = ( conductivity_tensor @ nu_N - torkivity_tensor @ nu_F)
        torkivity          = (-conductivity_tensor @ nu_F + torkivity_tensor @ nu_N)
        avg_conductivity   = (conductivity_tensor + torkivity_tensor)   @ (nu_N - nu_F)
        delta_conductivity = (conductivity_tensor - torkivity_tensor) @ (nu_N + nu_F)
        torkivity_mag      = torkivity_tensor_mag @ (nu_N + nu_F)
        if flip:
            torkivity_mag *= -1
        
        self.conductance        = OrtTransf.T @ conductance_tensor      @ OrtTransf
        self.torkance           = OrtTransf.T @ torkance_tensor         @ OrtTransf
        self.conductivity       = OrtTransf.T @ conductivity      
        self.torkivity          = OrtTransf.T @ torkivity         
        self.avg_conductance    = OrtTransf.T @ (conductance_tensor + torkance_tensor) @ OrtTransf
        self.delta_conductance  = OrtTransf.T @ (conductance_tensor - torkance_tensor) @ OrtTransf
        self.torkance_mag       = OrtTransf.T @ torkance_tensor_mag    @ OrtTransf
        self.avg_conductivity   = OrtTransf.T @ avg_conductivity
        self.delta_conductivity = OrtTransf.T @ delta_conductivity
        self.torkivity_mag      = OrtTransf.T @ torkivity_mag


    # def old__init__old(self,m,E,beta_S,int_params,flip=False):
    #     OrtTransf = self.Lab_frame_orthogonal_transformation(E,flip)
    #     self.m = OrtTransf[:3,:3] @ m
    #     self.beta_S = beta_S
    #     self.u0  = int_params.u0
    #     self.uEx = int_params.uEx
    #     self.uR  = int_params.uR
    #     self.uD  = int_params.uD
    #     self.kF  = int_params.kF
    #     self.vF  = hbar_*int_params.kF/me_
    #     self.t_NM = int_params.t_NM
    #     self.t_FM = int_params.t_FM
    #     self.mu_approx = int_params.mu_approx
    #     self.theta, self.phi, self.weight = int_params.theta, int_params.phi, int_params.weight
    #     self.conductance, self.torkance, self.avg_conductance, self.delta_conductance, self.torkance_mag, conductivity_tensor, torkivity_tensor, avg_conductivity_tensor, delta_conductivity_tensor, torkivity_tensor_mag = self.Compute_scattering_matricies()
        
    #     # shifted distributions
    #     # NM
    #     q_m = np.zeros(4)
    #     q_m[3] = 1.0
    #     q_m *= self.vF * self.t_NM

    #     # FM
    #     q_p = np.zeros(4)
    #     q_p[0] = -self.beta_S * self.m[0]
    #     q_p[1] = -self.beta_S * self.m[1]
    #     q_p[2] = -self.beta_S * self.m[2]
    #     q_p[3] = 1.0
    #     q_p *= self.vF * self.t_FM

    #     # difference and average
    #     q_delta = (q_m - q_p)/2
    #     q_avg   = (q_m + q_p)/2
    
    #     self.conductivity       = ( conductivity_tensor @ q_m - torkivity_tensor @ q_p)
    #     self.torkivity          = (-conductivity_tensor @ q_p + torkivity_tensor @ q_m)
    #     self.avg_conductivity   = avg_conductivity_tensor   @ q_delta
    #     self.delta_conductivity = delta_conductivity_tensor @ q_avg
    #     self.torkivity_mag      = 2*torkivity_tensor_mag @ q_avg
        
    #     self.conductance        = OrtTransf.T @ self.conductance       @ OrtTransf
    #     self.torkance           = OrtTransf.T @ self.torkance          @ OrtTransf
    #     self.conductivity       = OrtTransf.T @ self.conductivity      
    #     self.torkivity          = OrtTransf.T @ self.torkivity         
    #     self.avg_conductance    = OrtTransf.T @ self.avg_conductance   @ OrtTransf
    #     self.delta_conductance  = OrtTransf.T @ self.delta_conductance @ OrtTransf
    #     self.torkance_mag       = OrtTransf.T @ self.torkance_mag      @ OrtTransf*2
    #     self.avg_conductivity   = OrtTransf.T @ self.avg_conductivity
    #     self.delta_conductivity = OrtTransf.T @ self.delta_conductivity
    #     self.torkivity_mag      = OrtTransf.T @ self.torkivity_mag*2

    def Gk_up_up(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)
        return np.cos(theta)**2/denominator
    
    def Gk_down_down(self,theta,u_up,u_down):
        denominator = (u_down**2 + np.cos(theta)**2)
        return np.cos(theta)**2/denominator

    def Gk_plus(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)*(u_down**2 + np.cos(theta)**2)
        return np.cos(theta)**2*(u_down**2 + 2*np.cos(theta)**2 + u_up**2)/denominator
    
    def Gk_minus(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)*(u_down**2 + np.cos(theta)**2)
        return np.cos(theta)**2*(u_down**2 - u_up**2)/denominator
    
    def Re_Gk_up_down(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)*(u_down**2 + np.cos(theta)**2)
        return np.cos(theta)**2*(u_down**2 + np.cos(theta)**2 - u_down*u_up +u_up**2)/denominator
    
    def Im_Gk_up_down(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)*(u_down**2 + np.cos(theta)**2)
        return np.cos(theta)*u_up*u_down*(u_up - u_down)/denominator
    
    def Re_Tk_up_down(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)*(u_down**2 + np.cos(theta)**2)
        return np.cos(theta)**2*(np.cos(theta)**2 + u_up*u_down)/denominator
    
    def Im_Tk_up_down(self,theta,u_up,u_down):
        denominator = (u_up**2 + np.cos(theta)**2)*(u_down**2 + np.cos(theta)**2)
        return np.cos(theta)**3*(u_down - u_up)/denominator
    
    def IR_matrix(self,theta_i,u_up,u_down,u):
        IR = np.zeros((4,4))
        IR += self.Gk_plus(theta_i,u_up,u_down)*self.u_outer_u(u)
        IR -= self.Gk_minus(theta_i,u_up,u_down)*self.u_sc_u_sc(u)
        IR += 2*self.Re_Gk_up_down(theta_i,u_up,u_down)*self.u_Cross_u_Cross(u)
        IR -= 2*self.Im_Gk_up_down(theta_i,u_up,u_down)*self.u_Cross(u)
        return IR
    
    def T_matrix(self,theta_i,u_up,u_down,u):
        T = np.zeros((4,4))
        T += self.Gk_plus(theta_i,u_up,u_down)*self.u_outer_u(u)
        T -= self.Gk_minus(theta_i,u_up,u_down)*self.u_sc_u_sc(u)
        T += 2*self.Re_Tk_up_down(theta_i,u_up,u_down)*self.u_Cross_u_Cross(u)
        T -= 2*self.Im_Tk_up_down(theta_i,u_up,u_down)*self.u_Cross(u)
        return T

    def norm_uEff(self,m,theta,phi):
        uEff = np.zeros(3)
        uEff[0] =  m[0]*self.uEx + np.sin(phi)*np.sin(theta)*self.uR + np.cos(phi)*np.sin(theta)*self.uD
        uEff[1] =  m[1]*self.uEx - np.cos(phi)*np.sin(theta)*self.uR - np.sin(phi)*np.sin(theta)*self.uD
        uEff[2] =  m[2]*self.uEx 
        return np.sqrt(np.dot(uEff,uEff))
    
    def uEff(self,m,theta,phi):
        uEff = np.zeros(3)
        uEff[0] =  m[0]*self.uEx + np.sin(phi)*np.sin(theta)*self.uR + np.cos(phi)*np.sin(theta)*self.uD
        uEff[1] =  m[1]*self.uEx - np.cos(phi)*np.sin(theta)*self.uR - np.sin(phi)*np.sin(theta)*self.uD
        uEff[2] =  m[2]*self.uEx 
        return uEff
    
    def Lab_frame_orthogonal_transformation(self,E,flip):
        # Omat = np.zeros((4,4))
        # # interface normal
        # n_hat = np.zeros(3)
        # n_hat[2] =  1
        # if flip:
        #     n_hat[2] = -1
        # # in-plane electric field direction
        # E_hat = E / np.linalg.norm(E)
        # # in-plane current normal direction
        # nxE_hat = np.cross(n_hat,E_hat)
        # # Orthogonal transformation matrix
        # Omat[0,:3] = E_hat
        # Omat[1,:3] = nxE_hat
        # Omat[2,:3] = n_hat # interface in z direction
        # Omat[3,3] = 1 # current in z direction
        # # if flip:
        # #     Omat *= -1
        Omat = np.identity(4)
        if flip:
            Omat[2,2] = 1
            Omat[3,3] = 1
        return Omat
    
    def u_outer_u(self,u):
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
        ux = np.zeros((4,4))
        ux[3,0] = u[0]
        ux[3,1] = u[1]
        ux[3,2] = u[2]
        ux[0,3] = u[0]
        ux[1,3] = u[1]
        ux[2,3] = u[2]
        return ux
    
    def mcross(self):
        # [m]_x matrix
        mx = np.zeros((4,4))
        mx[0,1] = -self.m[2]
        mx[0,2] =  self.m[1]
        mx[1,0] =  self.m[2]
        mx[1,2] = -self.m[0]
        mx[2,0] = -self.m[1]
        mx[2,1] =  self.m[0]
        return mx
    
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
    
    def Compute_scalar_conductances(self,u0,uex,Rashba=False):
        G_0_up_up      = 0
        G_0_down_down  = 0
        Re_G_0_up_down = 0
        Im_G_0_up_down = 0
        Re_T_0_up_down = 0
        Im_T_0_up_down = 0
        Re_U_0_up_down = 0
        Im_U_0_up_down = 0
        G_1_up_up      = 0
        G_1_down_down  = 0
        Im_G_1_up_down = 0
        Im_T_1_up_down = 0
        # Compute quadrature
        t0 = time.time()
        for i in range(len(self.theta)):
            # compute chi_up and chi_down
            if Rashba:
                u_up   = u0  - uex*abs(np.sin(self.theta[i]))# majority
                u_down = u0  + uex*abs(np.sin(self.theta[i])) # minority
            else:
                u_up   = u0 - uex
                u_down = u0 + uex

            w_i  = 4*np.pi*self.weight[i]
            kz_i = np.cos(self.theta[i])
            kx_i = np.cos(self.phi[i])*np.sin(self.theta[i])
            if Rashba:
                bxb = np.cos(self.phi[i])**2
                b = np.cos(self.phi[i])
            else:
                bxb = 1
                b = 1

            # Set_up the scattering matricies for the current quadrature point multiply by weights & add to sum over quadrature points
            # conductance tensors
            G_0_up_up      += w_i*kz_i*self.Gk_up_up(self.theta[i],u_up,u_down)*bxb
            G_0_down_down  += w_i*kz_i*self.Gk_down_down(self.theta[i],u_up,u_down)*bxb
            Re_G_0_up_down += w_i*kz_i*self.Re_Gk_up_down(self.theta[i],u_up,u_down)*bxb
            Im_G_0_up_down += w_i*kz_i*self.Im_Gk_up_down(self.theta[i],u_up,u_down)*bxb
            Re_T_0_up_down += w_i*kz_i*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*bxb
            Im_T_0_up_down += w_i*kz_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*bxb

            # spin & charge density tensors
            if Rashba == False:
                Re_U_0_up_down += 2*uex*w_i*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*bxb
                Im_U_0_up_down += 2*uex*w_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*bxb
         
            # shifted conductance tensors
            G_1_up_up      += w_i*kx_i*kz_i*(self.Gk_up_up(self.theta[i],u_up,u_down))*b
            G_1_down_down  += w_i*kx_i*kz_i*self.Gk_down_down(self.theta[i],u_up,u_down)*b
            Im_G_1_up_down += w_i*kx_i*kz_i*self.Im_Gk_up_down(self.theta[i],u_up,u_down)*b
            Im_T_1_up_down += w_i*kx_i*kz_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*b
        t1 = time.time()
        total = t1-t0
        print('Quadrature computation time: ',total)

        # Scale by constants
        G_0_up_up      *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        G_0_down_down  *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_G_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_G_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_T_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_T_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)

        Re_U_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_U_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)

        G_1_up_up      *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        G_1_down_down  *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_G_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_T_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        return G_0_up_up, G_0_down_down, Re_G_0_up_down, Im_G_0_up_down, Re_T_0_up_down, Im_T_0_up_down, Re_U_0_up_down, Im_U_0_up_down, G_1_up_up, G_1_down_down, Im_G_1_up_down, Im_T_1_up_down
    
    def Compute_scattering_matricies(self):
        G_0_plus       = np.zeros((4,4))
        G_0_minus      = np.zeros((4,4))
        Re_G_0_up_down = np.zeros((4,4))
        Im_G_0_up_down = np.zeros((4,4))
        Re_T_0_up_down = np.zeros((4,4))
        Im_T_0_up_down = np.zeros((4,4))
        U_0_Plus       = np.zeros((4,4))
        U_0_minus      = np.zeros((4,4))
        Re_U_0_up_down = np.zeros((4,4))
        Im_U_0_up_down = np.zeros((4,4))
        # shifted conductance matricies
        G_1_plus       = np.zeros((4,4))
        G_1_minus      = np.zeros((4,4))
        Re_G_1_up_down = np.zeros((4,4))
        Im_G_1_up_down = np.zeros((4,4))
        Re_T_1_up_down = np.zeros((4,4))
        Im_T_1_up_down = np.zeros((4,4))
        U_1_Plus       = np.zeros((4,4))
        U_1_minus      = np.zeros((4,4))
        Re_U_1_up_down = np.zeros((4,4))
        Im_U_1_up_down = np.zeros((4,4))
        # Compute quadrature
        t0 = time.time()
        for i in range(len(self.theta)):
            # compute chi_up and chi_down
            u_up   = self.u0 - self.norm_uEff(self.m,self.theta[i],self.phi[i]) # majority
            u_down = self.u0 + self.norm_uEff(self.m,self.theta[i],self.phi[i]) # minority
            if(self.norm_uEff(self.m,self.theta[i],self.phi[i]) > 0.0):
                b_vec = self.uEff(self.m,self.theta[i],self.phi[i])/self.norm_uEff(self.m,self.theta[i],self.phi[i])
            else:
                b_vec = np.zeros(3)

            w_i  = 4*np.pi*self.weight[i]
            kz_i = np.cos(self.theta[i])
            kx_i = np.cos(self.phi[i])*np.sin(self.theta[i])

            # Set_up the scattering matricies for the current quadrature point multiply by weights & add to sum over quadrature points
            # conductance tensors
            G_0_plus       += w_i*kz_i*self.Gk_plus(self.theta[i],u_up,u_down)*self.u_outer_u(b_vec)
            G_0_minus      += w_i*kz_i*self.Gk_minus(self.theta[i],u_up,u_down)*self.u_sc_u_sc(b_vec)
            Re_G_0_up_down += w_i*kz_i*self.Re_Gk_up_down(self.theta[i],u_up,u_down)*self.u_Cross_u_Cross(b_vec)
            Im_G_0_up_down += w_i*kz_i*self.Im_Gk_up_down(self.theta[i],u_up,u_down)*self.u_Cross(b_vec)
            Re_T_0_up_down += w_i*kz_i*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross_u_Cross(b_vec)
            Im_T_0_up_down += w_i*kz_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross(b_vec)
            # spin & charge density tensors
            U_0_Plus       += w_i*self.Gk_plus(self.theta[i],u_up,u_down)*self.u_outer_u(b_vec)
            U_0_minus      += w_i*self.Gk_minus(self.theta[i],u_up,u_down)*self.u_sc_u_sc(b_vec)
            Re_U_0_up_down += w_i*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross_u_Cross(b_vec)
            Im_U_0_up_down += w_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross(b_vec)

            # shifted conductance tensors
            G_1_plus       += w_i*kx_i*kz_i*self.Gk_plus(self.theta[i],u_up,u_down)*self.u_outer_u(b_vec)
            G_1_minus      += w_i*kx_i*kz_i*self.Gk_minus(self.theta[i],u_up,u_down)*self.u_sc_u_sc(b_vec)
            Re_G_1_up_down += w_i*kx_i*kz_i*self.Re_Gk_up_down(self.theta[i],u_up,u_down)*self.u_Cross_u_Cross(b_vec)
            Im_G_1_up_down += w_i*kx_i*kz_i*self.Im_Gk_up_down(self.theta[i],u_up,u_down)*self.u_Cross(b_vec)
            Re_T_1_up_down += w_i*kx_i*kz_i*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross_u_Cross(b_vec)
            Im_T_1_up_down += w_i*kx_i*kz_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross(b_vec)
            # shifted spin & charge density tensors
            U_1_Plus       += w_i*kx_i*self.Gk_plus(self.theta[i],u_up,u_down)*self.u_outer_u(b_vec)
            U_1_minus      += w_i*kx_i*self.Gk_minus(self.theta[i],u_up,u_down)*self.u_sc_u_sc(b_vec)
            Re_U_1_up_down += w_i*kx_i*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross_u_Cross(b_vec)
            Im_U_1_up_down += w_i*kx_i*self.Im_Tk_up_down(self.theta[i],u_up,u_down)*self.u_Cross(b_vec)
        t1 = time.time()
        total = t1-t0
        print('Quadrature computation time: ',total)

        # Scale by constants
        G_0_plus       *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        G_0_minus      *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_G_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_G_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_T_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_T_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        U_0_Plus       *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        U_0_minus      *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_U_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_U_0_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        G_1_plus       *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        G_1_minus      *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_G_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_G_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_T_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_T_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        U_1_Plus       *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        U_1_minus      *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Re_U_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        Im_U_1_up_down *= ee_**2*self.kF**2/(hbar_*8*np.pi**3)

        return G_0_plus, G_0_minus, Re_G_0_up_down, Im_G_0_up_down, Re_T_0_up_down, Im_T_0_up_down, U_0_Plus, U_0_minus, Re_U_0_up_down, Im_U_0_up_down, G_1_plus, G_1_minus, Re_G_1_up_down, Im_G_1_up_down, Re_T_1_up_down, Im_T_1_up_down, U_1_Plus, U_1_minus, Re_U_1_up_down, Im_U_1_up_down
    
    
    def Compute_scattering_matricies_old(self):
        # initialize scattering matricies
        B_IR = np.zeros((4,4))
        B_T  = np.zeros((4,4))
        A_IR = np.zeros((4,4))
        A_T = np.zeros((4,4))
        b_IR = np.zeros((4,4))
        b_T = np.zeros((4,4))
        a_IR = np.zeros((4,4))
        a_T = np.zeros((4,4))
        # Compute quadrature
        t0 = time.time()
        for i in range(len(self.theta)):
            # compute chi_up and chi_down
            u_up   = self.u0 - self.norm_uEff(self.m,self.theta[i],self.phi[i]) # majority
            u_down = self.u0 + self.norm_uEff(self.m,self.theta[i],self.phi[i]) # minority
            if(self.norm_uEff(self.m,self.theta[i],self.phi[i]) > 0.0):
                u = self.uEff(self.m,self.theta[i],self.phi[i])/self.norm_uEff(self.m,self.theta[i],self.phi[i])
            else:
                u = np.zeros(3)

            # compute scattering matricies
            IR = self.IR_matrix(self.theta[i],u_up,u_down,u)
            T = self.T_matrix(self.theta[i],u_up,u_down,u)

            s0_ir = 2*np.identity(4) - IR.copy() # convert to I+R
            s0_t  = T.copy()
            S0_IR = IR.copy()
            S0_T  = T.copy()
            
            s1_ir = 2*np.identity(4) - IR.copy() # convert to I+R
            s1_t  = T.copy()
            S1_IR = IR.copy()
            S1_T  = T.copy()
            
            # in-plane charge currents driving out-of-plane spin currents
            # Todo: findout if sin(theta) is needed
            s1_ir *= np.cos(self.phi[i])*np.sin(self.theta[i])
            s1_t  *= np.cos(self.phi[i])*np.sin(self.theta[i])
            S1_IR *= np.cos(self.phi[i])*np.sin(self.theta[i])
            S1_T  *= np.cos(self.phi[i])*np.sin(self.theta[i])

            # Multiply by weights
            S0_IR *= self.weight[i]*np.cos(self.theta[i])
            S0_T  *= self.weight[i]*np.cos(self.theta[i])
            s0_ir *= self.weight[i]
            s0_t  *= self.weight[i]
            S1_IR *= self.weight[i]*np.cos(self.theta[i])
            S1_T  *= self.weight[i]*np.cos(self.theta[i])
            s1_ir *= self.weight[i]
            s1_t  *= self.weight[i]
            
            # sum over quadrature points
            B_IR    += S0_IR
            B_T     += S0_T
            A_IR    += s0_ir
            A_T     += s0_t
            b_IR    += S1_IR
            b_T     += S1_T
            a_IR    += s1_ir
            a_T     += s1_t
        t1 = time.time()
        total = t1-t0
        print('Quadrature computation time: ',total)

        B_IR *= 4*np.pi
        B_T  *= 4*np.pi
        A_IR *= 4*np.pi
        A_T  *= 4*np.pi
        b_IR *= 4*np.pi
        b_T  *= 4*np.pi
        a_IR *= 4*np.pi
        a_T  *= 4*np.pi

        conductance        = B_IR
        torkance           = B_T
        conductivity       = b_IR
        torkivity          = b_T
        if(self.mu_approx): # assume the incoming spin currents behave as if they originate from spin-dependent reservoirs
            avg_conductance    = B_IR + B_T
            delta_conductance  = B_IR - B_T
            torkance_mag       = self.mcross()@A_T
            avg_conductivity   = b_IR + b_T
            delta_conductivity = b_IR - b_T
            torkivity_mag      = self.mcross()@a_T
        else: # spin current tensors in terms of accumulation  tensors
            avg_conductance    = (B_IR + B_T)@np.linalg.inv(A_IR - A_T)
            delta_conductance  = (B_IR - B_T)@np.linalg.inv(A_IR + A_T)
            torkance_mag       = (self.mcross()@A_T)@np.linalg.inv(A_T + A_IR)
            avg_conductivity   = (b_IR + b_T) - avg_conductance@(a_IR - a_T)
            delta_conductivity = (b_IR - b_T) - delta_conductance@(a_IR + a_T)
            torkivity_mag = self.mcross()@a_T - torkance_mag@(a_T + a_IR)
        # Multiply by constants 
        conductance        *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        torkance           *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        conductivity       *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        torkivity          *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        avg_conductance    *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        delta_conductance  *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        torkance_mag       *= self.uEx*ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        avg_conductivity   *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        delta_conductivity *=          ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        torkivity_mag      *= self.uEx*ee_**2*self.kF**2/(hbar_*8*np.pi**3)
        return conductance, torkance, avg_conductance, delta_conductance, torkance_mag, conductivity, torkivity, avg_conductivity, delta_conductivity, torkivity_mag

    def Compute_conductances(self):
            # Computes the individual conductance contributions 
            # initialize scattering matricies
            G0_minus_m = 0.0
            G0_plus_mm = 0.0
            G0_plus_mR = np.zeros(3)
            G0_minus_R = np.zeros(3)
            G0_plus_RR = np.zeros((3,3))
            G0_mix_m   = 0.0
            G0_mix_mm  = 0.0
            G0_mix_mR  = np.zeros(3)
            G0_mix_R   = np.zeros(3)
            G0_mix_RR  = np.zeros((3,3))
            T0_mix_m   = 0.0
            T0_mix_mm  = 0.0
            T0_mix_mR  = np.zeros(3)
            T0_mix_R   = np.zeros(3)
            T0_mix_RR  = np.zeros((3,3))
            U0_plus_mR = np.zeros(3)
            U0_minus_R = np.zeros(3)
            U0_plus_RR = np.zeros((3,3))
            U0_mix_mR  = np.zeros(3)
            U0_mix_R   = np.zeros(3)
            U0_mix_RR  = np.zeros((3,3))

            G1_minus_m = 0.0
            G1_plus_mm = 0.0
            G1_plus_mR = np.zeros(3)
            G1_minus_R = np.zeros(3)
            G1_plus_RR = np.zeros((3,3))
            G1_mix_m   = 0.0
            G1_mix_mm  = 0.0
            G1_mix_mR  = np.zeros(3)
            G1_mix_R   = np.zeros(3)
            G1_mix_RR  = np.zeros((3,3))
            T1_mix_m   = 0.0
            T1_mix_mm  = 0.0
            T1_mix_mR  = np.zeros(3)
            T1_mix_R   = np.zeros(3)
            T1_mix_RR  = np.zeros((3,3))
            U1_plus_mR = np.zeros(3)
            U1_minus_R = np.zeros(3)
            U1_plus_RR = np.zeros((3,3))
            U1_mix_mR  = np.zeros(3)
            U1_mix_R   = np.zeros(3)
            U1_mix_RR  = np.zeros((3,3))
            # Compute quadrature
            for i in range(len(self.theta)):
                # compute chi_up and chi_down
                u_up   = self.u0 - self.norm_uEff(self.m,self.theta[i],self.phi[i]) # majority
                u_down = self.u0 + self.norm_uEff(self.m,self.theta[i],self.phi[i]) # minority
                if(self.norm_uEff(self.m,self.theta[i],self.phi[i]) > 0.0):
                    u = self.uEff(self.m,self.theta[i],self.phi[i])/self.norm_uEff(self.m,self.theta[i],self.phi[i])
                else:
                    u = np.zeros(3)

                # rashba field
                kxz_vec = self.uEff(np.zeros(3),self.theta[i],self.phi[i])
                kxz_outer_kxz = self.u_outer_u(kxz_vec)[:3,:3]
                G0 = ee_**2*self.kF**2/(hbar_*8*np.pi**3)
                if(self.norm_uEff(self.m,self.theta[i],self.phi[i])> 0.0):
                    inv_unorm = 1.0/self.norm_uEff(self.m,self.theta[i],self.phi[i])
                else:
                    inv_unorm = 0.0

                G0_minus_m += 4*np.pi*G0*self.uEx         *self.Gk_minus(self.theta[i],u_up,u_down)*inv_unorm                             *self.weight[i]*np.cos(self.theta[i])
                G0_plus_mm += 4*np.pi*G0*self.uEx*self.uEx*self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm                    *self.weight[i]*np.cos(self.theta[i])
                G0_plus_mR += 4*np.pi*G0*self.uEx         *self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_vec      *self.weight[i]*np.cos(self.theta[i])
                G0_minus_R += 4*np.pi*G0                  *self.Gk_minus(self.theta[i],u_up,u_down)*inv_unorm               *kxz_vec      *self.weight[i]*np.cos(self.theta[i])
                G0_plus_RR += 4*np.pi*G0                  *self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_outer_kxz*self.weight[i]*np.cos(self.theta[i])
                G0_mix_m   += 4*np.pi*G0*self.uEx         *self.Im_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm                        *self.weight[i]*np.cos(self.theta[i])
                G0_mix_mm  += 4*np.pi*G0*self.uEx*self.uEx*self.Re_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm              *self.weight[i]*np.cos(self.theta[i])
                G0_mix_mR  += 4*np.pi*G0*self.uEx         *self.Re_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_vec      *self.weight[i]*np.cos(self.theta[i])
                G0_mix_R   += 4*np.pi*G0                  *self.Im_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm          *kxz_vec      *self.weight[i]*np.cos(self.theta[i])
                G0_mix_RR  += 4*np.pi*G0                  *self.Re_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_outer_kxz*self.weight[i]*np.cos(self.theta[i])
                T0_mix_m   += 4*np.pi*G0*self.uEx         *self.Im_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm                        *self.weight[i]*np.cos(self.theta[i])
                T0_mix_mm  += 4*np.pi*G0*self.uEx*self.uEx*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm              *self.weight[i]*np.cos(self.theta[i])
                T0_mix_mR  += 4*np.pi*G0*self.uEx         *self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_vec      *self.weight[i]*np.cos(self.theta[i])
                T0_mix_R   += 4*np.pi*G0                  *self.Im_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm          *kxz_vec      *self.weight[i]*np.cos(self.theta[i])
                T0_mix_RR  += 4*np.pi*G0                  *self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_outer_kxz*self.weight[i]*np.cos(self.theta[i])
                U0_plus_mR += 4*np.pi*G0*self.uEx*self.uEx*self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_vec      *self.weight[i]
                U0_minus_R += 4*np.pi*G0*self.uEx         *self.Gk_minus(self.theta[i],u_up,u_down)*inv_unorm               *kxz_vec      *self.weight[i]
                U0_plus_RR += 4*np.pi*G0*self.uEx         *self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_outer_kxz*self.weight[i]
                U0_mix_mR  += 4*np.pi*G0*self.uEx*self.uEx*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_vec      *self.weight[i]
                U0_mix_R   += 4*np.pi*G0*self.uEx         *self.Im_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm          *kxz_vec      *self.weight[i]
                U0_mix_RR  += 4*np.pi*G0*self.uEx         *self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_outer_kxz*self.weight[i]
                
                # in-plane charge currents driving out-of-plane spin currents
                G1_minus_m += 4*np.pi*G0*self.uEx         *self.Gk_minus(self.theta[i],u_up,u_down)*inv_unorm                             *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_plus_mm += 4*np.pi*G0*self.uEx*self.uEx*self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm                    *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_plus_mR += 4*np.pi*G0*self.uEx         *self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_vec      *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_minus_R += 4*np.pi*G0                  *self.Gk_minus(self.theta[i],u_up,u_down)*inv_unorm               *kxz_vec      *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_plus_RR += 4*np.pi*G0                  *self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_outer_kxz*self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_mix_m   += 4*np.pi*G0*self.uEx         *self.Im_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm                        *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_mix_mm  += 4*np.pi*G0*self.uEx*self.uEx*self.Re_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm              *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_mix_mR  += 4*np.pi*G0*self.uEx         *self.Re_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_vec      *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_mix_R   += 4*np.pi*G0                  *self.Im_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm          *kxz_vec      *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                G1_mix_RR  += 4*np.pi*G0                  *self.Re_Gk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_outer_kxz*self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                T1_mix_m   += 4*np.pi*G0*self.uEx         *self.Im_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm                        *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                T1_mix_mm  += 4*np.pi*G0*self.uEx*self.uEx*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm              *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                T1_mix_mR  += 4*np.pi*G0*self.uEx         *self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_vec      *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                T1_mix_R   += 4*np.pi*G0                  *self.Im_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm          *kxz_vec      *self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                T1_mix_RR  += 4*np.pi*G0                  *self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_outer_kxz*self.weight[i]*np.cos(self.theta[i])*np.cos(self.phi[i])*np.sin(self.theta[i])
                U1_plus_mR += 4*np.pi*G0*self.uEx*self.uEx*self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_vec      *self.weight[i]                      *np.cos(self.phi[i])*np.sin(self.theta[i])
                U1_minus_R += 4*np.pi*G0*self.uEx         *self.Gk_minus(self.theta[i],u_up,u_down)*inv_unorm               *kxz_vec      *self.weight[i]                      *np.cos(self.phi[i])*np.sin(self.theta[i])
                U1_plus_RR += 4*np.pi*G0*self.uEx         *self.Gk_plus(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm      *kxz_outer_kxz*self.weight[i]                      *np.cos(self.phi[i])*np.sin(self.theta[i])
                U1_mix_mR  += 4*np.pi*G0*self.uEx*self.uEx*self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_vec      *self.weight[i]                      *np.cos(self.phi[i])*np.sin(self.theta[i])
                U1_mix_R   += 4*np.pi*G0*self.uEx         *self.Im_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm          *kxz_vec      *self.weight[i]                      *np.cos(self.phi[i])*np.sin(self.theta[i])
                U1_mix_RR  += 4*np.pi*G0*self.uEx         *self.Re_Tk_up_down(self.theta[i],u_up,u_down)*inv_unorm*inv_unorm*kxz_outer_kxz*self.weight[i]                      *np.cos(self.phi[i])*np.sin(self.theta[i])
        
            return G0_minus_m, G0_plus_mm, G0_plus_mR, G0_minus_R, G0_plus_RR, G0_mix_m, G0_mix_mm, G0_mix_mR, G0_mix_R, G0_mix_RR, T0_mix_m, T0_mix_mm, T0_mix_mR, T0_mix_R, T0_mix_RR, U0_plus_mR, U0_minus_R, U0_plus_RR, U0_mix_mR, U0_mix_R, U0_mix_RR, G1_minus_m, G1_plus_mm, G1_plus_mR, G1_minus_R, G1_plus_RR, G1_mix_m, G1_mix_mm, G1_mix_mR, G1_mix_R, G1_mix_RR, T1_mix_m, T1_mix_mm, T1_mix_mR, T1_mix_R, T1_mix_RR, U1_plus_mR, U1_minus_R, U1_plus_RR, U1_mix_mR, U1_mix_R, U1_mix_RR