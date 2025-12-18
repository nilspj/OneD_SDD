import numpy as np 
from .bulk_equations import FM_layer, NM_layer, HM_layer, AFM_layer, HFM_layer
from .interface_equations import ContinousBoundaryCondition, RashbaSpinMixingBoundaryCondition,RashbaPerturbSpinMixingBoundaryCondition, Magnetoelectronic_circuit_theory, Magnetoelectronic_circuit_theory_with_Rashba
from .physical_constants import mub_,ee_

class SDD_1D_solver:
    """
    Class representing a linear system for spin diffusion and spin current calculations in a layered structure.
    
    Attributes:
        geometry (object): The geometry of the layered structure.
        bulk_parameters (object): The bulk parameters of the materials in the layered structure.
        n_interfaces (int): The number of interfaces in the layered structure.
        n_materials (int): The number of materials in the layered structure.
        layers (ndarray): Array of layer objects representing the materials in the layered structure.
        M (ndarray): Spin current boundary matrix.
        b (ndarray): Boundary condition vector.
        x (ndarray): Solution vector.
    """
    
    def __init__(self,E, geometry, bulk_parameters, int_parameters):
        """
        Initializes a new instance of the SDD_1D_solver class.
        
        Args:
            geometry (object): The geometry of the layered structure.
            bulk_parameters (object): The bulk parameters of the materials in the layered structure.
        
        Raises:
            ValueError: If the number of interfaces and materials do not match.
        """
        self.E = E
        self.geometry = geometry
        self.stack_thickness = abs(geometry[0] - geometry[-1])
        self.n_interfaces = len(geometry) # number of interfaces
        self.n_materials = len(bulk_parameters) # number of materials
        if self.n_interfaces != self.n_materials + 1:
            raise ValueError('Error: number of interfaces and materials do not match')
        self.bulk_parameters = bulk_parameters
        self.int_parameters = int_parameters
        if self.n_interfaces-2 != len(int_parameters):
            raise ValueError('Error: number of interfaces and interface parameters do not match')
        self.__assemble_layers()
        self.__assemble_internal_interfaces()
        self.M = np.zeros((0, 8 * self.n_materials), dtype=np.complex128) # spin current boundary matrix
        self.b = np.zeros(0, dtype=np.complex128)                      # boundary condition vector
        self.x = np.zeros(8 * self.n_materials, dtype=np.complex128)     # solution vector
        self.__assemble_linear_system()
        pass

    def __assemble_layers(self):
        """
        Assembles the layers of the layered structure based on the given geometry and bulk parameters.
        
        Args:
            geometry (object): The geometry of the layered structure.
            bulk_parameters (object): The bulk parameters of the materials in the layered structure.
        
        Raises:
            ValueError: If the material is not recognized.
        """
        self.layers = np.zeros(self.n_materials, dtype="object")
        for i in range(self.n_materials):
            if self.bulk_parameters[i].type == 'FM':
                self.layers[i] = FM_layer( self.E,self.geometry[i], self.geometry[i + 1], self.bulk_parameters[i],self.stack_thickness)
            elif self.bulk_parameters[i].type == 'NM':
                self.layers[i] = NM_layer( self.E,self.geometry[i], self.geometry[i + 1], self.bulk_parameters[i],self.stack_thickness)
            elif self.bulk_parameters[i].type == 'HM':
                self.layers[i] = HM_layer( self.E,self.geometry[i], self.geometry[i + 1], self.bulk_parameters[i],self.stack_thickness)
            elif self.bulk_parameters[i].type == 'AFM':
                self.layers[i] = AFM_layer(self.E,self.geometry[i], self.geometry[i + 1], self.bulk_parameters[i],self.stack_thickness)
            elif self.bulk_parameters[i].type == 'HFM':
                self.layers[i]= HFM_layer(self.E,self.geometry[i], self.geometry[i + 1], self.bulk_parameters[i],self.stack_thickness)
            else:
                raise ValueError('Error: material not recognized')
        pass
    
    def __assemble_internal_interfaces(self):
        self.interfaces = np.zeros(self.n_interfaces -2, dtype="object")
        for i in range(self.n_interfaces - 2):
            if self.int_parameters[i].type == 'continous':
                self.interfaces[i] = ContinousBoundaryCondition(       self.n_materials,i+1, self.layers[i], self.layers[i+1])
            elif self.int_parameters[i].type == 'RashbaSpinMixing':
                self.interfaces[i] = RashbaSpinMixingBoundaryCondition(self.n_materials,i+1, self.layers[i], self.layers[i+1], self.int_parameters[i])
            elif self.int_parameters[i].type == 'RashbaPerturbSpinMixing':
                self.interfaces[i] = RashbaPerturbSpinMixingBoundaryCondition(self.n_materials,i+1, self.layers[i], self.layers[i+1], self.int_parameters[i])
            elif self.int_parameters[i].type == 'MCT':
                self.interfaces[i] = Magnetoelectronic_circuit_theory( self.n_materials,i+1, self.layers[i], self.layers[i+1], self.int_parameters[i])
            elif self.int_parameters[i].type == 'MCT_Rashba':
                self.interfaces[i] = Magnetoelectronic_circuit_theory_with_Rashba( self.n_materials,i+1, self.layers[i], self.layers[i+1], self.int_parameters[i])
            else:
                raise ValueError('Error: boundary condition not recognized')
        pass
            
    
    def get_linear_system(self):
        """
        Returns the spin current boundary matrix and the boundary condition vector.
        
        Returns:
            tuple: A tuple containing the spin current boundary matrix and the boundary condition vector.
        """
        return self.M, self.b

    def __assemble_linear_system(self):
        """
        Assembles the linear system for solving the spin diffusion and spin current equations.
        """
        for i in range(self.n_interfaces):
            if i == 0:
                # bottom boundary condition
                sub_M = np.zeros((4, 8 * self.n_materials), dtype=np.complex128)
                sub_b = np.zeros(4, dtype=np.complex128)
                sub_M[0:4, 0:8] -= self.layers[i].J_low_external_boundary_M()
                sub_b[0:4] -= self.layers[i].J_low_external_boundary_b()
                self.M = np.append(self.M, sub_M, axis=0)
                self.b = np.append(self.b, sub_b, axis=0)
            elif i == self.n_interfaces - 1:
                # top boundary condition
                sub_M = np.zeros((4, 8 * self.n_materials), dtype=np.complex128)
                sub_b = np.zeros(4, dtype=np.complex128)
                sub_M[0:4, 8 * self.n_materials - 8:8 * self.n_materials] += self.layers[i - 1].J_high_external_boundary_M()
                sub_b[0:4] += self.layers[i - 1].J_high_external_boundary_b()
                self.M = np.append(self.M, sub_M, axis=0)
                self.b = np.append(self.b, sub_b, axis=0)
            else:
                # internal boundary condition
                sub_M = np.zeros((8, 8 * self.n_materials), dtype=np.complex128)
                sub_b = np.zeros(8, dtype=np.complex128)

                BoundaryCondition = self.interfaces[i-1]
                sub_M = BoundaryCondition.assemble_M()
                sub_b = BoundaryCondition.assemble_b()
                
                self.M = np.append(self.M, sub_M, axis=0)
                self.b = np.append(self.b, sub_b, axis=0)
        pass

    def solve_linear_system(self):
        """
        Solves the linear system to obtain the solution vector.
        """
        self.x = np.linalg.solve(self.M, self.b)
        return   
    
    def get_solution(self):
        """
        Returns the solution vector.
        
        Returns:
            ndarray: The solution vector.
        """
        return self.x

    def get_spin_accumulation(self,scale = True):
        """
        Calculates and returns the spin accumulation in the layered structure.
        
        Returns:
            tuple: A tuple containing the z-coordinates and the spin accumulation components (Sx, Sy, Sz).
        """
        z_tot = np.zeros(0)
        Sx_tot = np.zeros(0)
        Sy_tot = np.zeros(0)
        Sz_tot = np.zeros(0)
        for i in range(self.n_materials):
            layer = self.layers[i]
            nms = abs(layer.z_1 - layer.z_2)/1e-9
            z = np.linspace(layer.z_1, layer.z_2, int(nms*100))
            A = self.x[i * 8:i * 8 + 4]
            B = self.x[i * 8 + 4:i * 8 + 8]
            Sx, Sy, Sz = layer.spin_accumulation(A, B, z, scale)
            z_tot = np.concatenate((z_tot, z))
            Sx_tot = np.concatenate((Sx_tot, Sx))
            Sy_tot = np.concatenate((Sy_tot, Sy))
            Sz_tot = np.concatenate((Sz_tot, Sz))
        return z_tot, Sx_tot, Sy_tot, Sz_tot

    def get_spin_current(self, scale = True):
        """
        Calculates and returns the spin current in the layered structure.

        returns:
            tuple: A tuple containing the z-coordinates and the spin current components (Jsx, Jsy, Jsz).
        """
        z_tot = np.zeros(0)
        Jsx_tot = np.zeros(0)
        Jsy_tot = np.zeros(0)
        Jsz_tot = np.zeros(0)
        for i in range(self.n_materials):
            layer = self.layers[i]
            nms = abs(layer.z_1 - layer.z_2)/1e-9
            z = np.linspace(layer.z_1, layer.z_2, int(nms*100))
            A = self.x[i * 8:i * 8 + 4]
            B = self.x[i * 8 + 4:i * 8 + 8]
            Jsx, Jsy, Jsz = layer.spin_current(A, B, z,scale)
            z_tot = np.concatenate((z_tot, z))
            Jsx_tot = np.concatenate((Jsx_tot, Jsx))
            Jsy_tot = np.concatenate((Jsy_tot, Jsy))
            Jsz_tot = np.concatenate((Jsz_tot, Jsz))
        return z_tot, Jsx_tot, Jsy_tot, Jsz_tot
    
     
    def get_charge_potential_and_current(self):
        """
        Calculates and returns the charge potential and charge current in the layered structure.
        
        Returns:
            tuple: A tuple containing the z-coordinates and the charge potential and charge current.
        """
        z_tot = np.zeros(0)
        V_tot = np.zeros(0)
        J_tot = np.zeros(0)
        for i in range(self.n_materials):
            layer = self.layers[i]
            nms = abs(layer.z_1 - layer.z_2)/1e-9
            z = np.linspace(layer.z_1, layer.z_2, int(nms*100))
            A = self.x[i * 8:i * 8 + 4]
            B = self.x[i * 8 + 4:i * 8 + 8]
            V, J = layer.charge_potential_and_current(A, B, z)
            z_tot = np.concatenate((z_tot, z))
            V_tot = np.concatenate((V_tot, V))
            J_tot = np.concatenate((J_tot, J))
        return z_tot, V_tot, J_tot
    
    
    def get_spin_torque(self,scale = True):
        """
        Calculates and returns the spin torque in the layered structure.
        
        Returns:
            tuple: A tuple containing the z-coordinates and the spin torque components (Tx, Ty, Tz).
        """
        z_tot = np.zeros(0)
        Tx_tot = np.zeros(0)
        Ty_tot = np.zeros(0)
        Tz_tot = np.zeros(0)
        for i in range(self.n_materials):
            # Adding bulk torque
            layer = self.layers[i]
            nms = abs(layer.z_1 - layer.z_2)/1e-9
            z = np.linspace(layer.z_1, layer.z_2, int(nms*100))
            if (layer.material == 'FM' or layer.material == 'HFM'):
                A = self.x[i * 8:i * 8 + 4]
                B = self.x[i * 8 + 4:i * 8 + 8]
                Tx, Ty, Tz = layer.spin_torque(A, B, z)
            else:
                Tx, Ty, Tz = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z))
            z_tot  = np.concatenate(( z_tot,  z))
            Tx_tot = np.concatenate((Tx_tot, Tx))
            Ty_tot = np.concatenate((Ty_tot, Ty))
            Tz_tot = np.concatenate((Tz_tot, Tz))
        for i in range(self.n_interfaces-2):
            # Adding interface torque
            if(self.int_parameters[i].type == 'RashbaSpinMixing'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])

                # Torkance contribution
                V_low = inter.lower_layer.charge_potential_and_current(A_low, B_low, z)[0]
                V_high = inter.upper_layer.charge_potential_and_current(A_high, B_high, z)[0]
                Tm = self.interfaces[i].spin_tensors.torkance_mag[:3,3]
                mu_low = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                mu_high = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = self.interfaces[i].spin_tensors.torkance_mag[:3,:3] + Tm*(V_high - V_low)
                Tx, Ty, Tz = torkance_mag @ (mu_high.flatten() + mu_low.flatten())
                if(self.interfaces[i].full_absorption):
                    # add bulk contribution
                    A = self.interfaces[i].transverse_transformation_tensor()
                    torkance  = A @ self.interfaces[i].T_tensor[:3,:3] 
                    conductance  = A @ self.interfaces[i].T_tensor[:3,:3] 
                    if(inter.lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        Tx_FM, Ty_FM, Tz_FM = conductance @ mu_high.flatten() - torkance @  mu_low.flatten()
                    else:
                        Tx_FM, Ty_FM, Tz_FM = conductance @ mu_low.flatten() - torkance @  mu_high.flatten()
                    Tx += Tx_FM
                    Ty += Ty_FM
                    Tz += Tz_FM

                # torkivity contribution
                torkivity_mag = self.interfaces[i].spin_tensors.torkivity_mag[:3]
                Tx += torkivity_mag[0]*abs(self.E[0])
                Ty += torkivity_mag[1]*abs(self.E[0])
                Tz += torkivity_mag[2]*abs(self.E[0])
                if(self.interfaces[i].full_absorption):
                    A = self.interfaces[i].transverse_transformation_tensor()
                    torkivity_FM  = A @ self.interfaces[i].gamma_tensor[:3]
                    if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        torkivity_FM*= -1
                    # add bulk contribution
                    Tx += torkivity_FM[0]*abs(self.E[0])
                    Ty += torkivity_FM[1]*abs(self.E[0])
                    Tz += torkivity_FM[2]*abs(self.E[0])
            
                ind = np.argwhere(z_tot == self.geometry[i+1]).flatten()
                ind.sort()
                if (inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    Tx_tot[ind[0] - 30:ind[-1]] += Tx/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Ty_tot[ind[0] - 30:ind[-1]] += Ty/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Tz_tot[ind[0] - 30:ind[-1]] += Tz/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                else:
                    Tx_tot[ind[0]+1:ind[0] + 31] += Tx/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Ty_tot[ind[0]+1:ind[0] + 31] += Ty/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Tz_tot[ind[0]+1:ind[0] + 31] += Tz/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])

            if(self.int_parameters[i].type == 'RashbaPerturbSpinMixing'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])

                # Torkance contribution
                mu_low = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                mu_high = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = self.interfaces[i].torkance_tensor()[:3,:3]
                Tx, Ty, Tz = torkance_mag @ (mu_high.flatten() + mu_low.flatten())
                Tx *=0
                Ty *=0
                Tz *=0
                if(self.interfaces[i].full_absorption):
                    # add bulk contribution
                    torkance_FM  = self.interfaces[i].MCT_transverse_transmission_tensor()[:3,:3]
                    Tx_FM, Ty_FM, Tz_FM = torkance_FM @ (mu_high.flatten() + mu_low.flatten())
                    Tx += Tx_FM
                    Ty += Ty_FM
                    Tz += Tz_FM
                # torkivity FM contribution
                if(self.interfaces[i].full_absorption):
                    torkivity_FM_E  = self.interfaces[i].torkivity_tensor()[:3]
                    if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        torkivity_FM_E*= -1
                    # add bulk contribution
                    Tx += torkivity_FM_E[0]
                    Ty += torkivity_FM_E[1]
                    Tz += torkivity_FM_E[2]

                ind = np.argwhere(z_tot == self.geometry[i+1]).flatten()
                ind.sort()
                if (inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    Tx_tot[ind[0] - 30:ind[-1]] += Tx/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Ty_tot[ind[0] - 30:ind[-1]] += Ty/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Tz_tot[ind[0] - 30:ind[-1]] += Tz/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                else:
                    Tx_tot[ind[0]+1:ind[0] + 31] += Tx/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Ty_tot[ind[0]+1:ind[0] + 31] += Ty/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Tz_tot[ind[0]+1:ind[0] + 31] += Tz/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])

            if(self.int_parameters[i].type == 'MCT'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])
                if(inter.upper_layer.material == 'FM'):
                    mu_NM = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                else:
                    mu_NM = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = np.real(self.interfaces[i].MCT_transverse_tensor()[:3,:3])
                Tx, Ty, Tz = torkance_mag @ mu_NM.flatten()

                ind = np.argwhere(z_tot == self.geometry[i+1]).flatten()
                ind.sort()
                if (inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    Tx_tot[ind[0] - 30:ind[-1]] += Tx/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Ty_tot[ind[0] - 30:ind[-1]] += Ty/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Tz_tot[ind[0] - 30:ind[-1]] += Tz/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                else:
                    Tx_tot[ind[0]+1:ind[0] + 31] += Tx/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Ty_tot[ind[0]+1:ind[0] + 31] += Ty/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Tz_tot[ind[0]+1:ind[0] + 31] += Tz/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])

            if(self.int_parameters[i].type == 'MCT_Rashba'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])

                # Torkance contribution
                mu_low = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                mu_high = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = self.interfaces[i].torkance_tensor()[:3,:3]
                Tx, Ty, Tz = torkance_mag @ (mu_high.flatten() + mu_low.flatten())
                if(self.interfaces[i].full_absorption):
                    # add bulk contribution
                    torkance_FM  = self.interfaces[i].MCT_transverse_transmission_tensor()[:3,:3]
                    Tx_FM, Ty_FM, Tz_FM = torkance_FM @ (mu_high.flatten() + mu_low.flatten())
                    Tx += Tx_FM
                    Ty += Ty_FM
                    Tz += Tz_FM
                # torkivity FM contribution
                if(self.interfaces[i].full_absorption):
                    torkivity_FM_E  = self.interfaces[i].torkivity_tensor()[:3]
                    if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        torkivity_FM_E*= -1
                    # add bulk contribution
                    Tx += torkivity_FM_E[0]
                    Ty += torkivity_FM_E[1]
                    Tz += torkivity_FM_E[2]

                ind = np.argwhere(z_tot == self.geometry[i+1]).flatten()
                ind.sort()
                if (inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    Tx_tot[ind[0] - 30:ind[-1]] += Tx/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Ty_tot[ind[0] - 30:ind[-1]] += Ty/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                    Tz_tot[ind[0] - 30:ind[-1]] += Tz/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                else:
                    Tx_tot[ind[0]+1:ind[0] + 31] += Tx/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Ty_tot[ind[0]+1:ind[0] + 31] += Ty/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                    Tz_tot[ind[0]+1:ind[0] + 31] += Tz/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                # if(self.interfaces[i].full_absorption):
                #     A_low = self.x[i * 8:i * 8 + 4]
                #     B_low = self.x[i * 8 + 4:i * 8 + 8]
                #     A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                #     B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                #     inter = self.interfaces[i]
                #     z  = np.array([self.geometry[i+1]])
                #     # torkance contribution
                #     if(inter.upper_layer.material == 'FM'):
                #         mu_NM = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                #     else:
                #         mu_NM = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                #     torkance_mag = np.real(self.interfaces[i].MCT_transverse_tensor()[:3,:3])
                #     Tx, Ty, Tz = torkance_mag @ mu_NM.flatten()
                    
                #     torkivity_FM_E  = self.interfaces[i].torkivity_tensor()[:3]
                #     if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                #         torkivity_FM_E*= -1
                #     # add bulk contribution
                #     Tx += torkivity_FM_E[0]
                #     Ty += torkivity_FM_E[1]
                #     Tz += torkivity_FM_E[2]

                #     ind = np.argwhere(z_tot == self.geometry[i+1]).flatten()
                #     ind.sort()
                #     if (inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                #         Tx_tot[ind[0] - 30:ind[-1]] += Tx/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                #         Ty_tot[ind[0] - 30:ind[-1]] += Ty/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                #         Tz_tot[ind[0] - 30:ind[-1]] += Tz/abs(z_tot[ind[0] - 30] - z_tot[ind[0]])
                #     else:
                #         Tx_tot[ind[0]+1:ind[0] + 31] += Tx/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                #         Ty_tot[ind[0]+1:ind[0] + 31] += Ty/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])
                #         Tz_tot[ind[0]+1:ind[0] + 31] += Tz/abs(z_tot[ind[0] + 31] - z_tot[ind[0]+1])

        # Convert from A/m^3 to A/ms 
        if scale:
            Tx_tot *= mub_/ee_
            Ty_tot *= mub_/ee_
            Tz_tot *= mub_/ee_
        return z_tot, Tx_tot, Ty_tot, Tz_tot
    
    def get_total_spin_torque(self,scale = True):
        """
        Calculates and returns the total spin torque in each layer of the layered structure.
        
        Returns:
            tuple: A tuple containing the torque integrated over each layer
        """
        T_tot = np.zeros((self.n_materials,3))
        for i in range(self.n_materials):
            layer = self.layers[i]
            nms = abs(layer.z_1 - layer.z_2)/1e-9
            z = np.linspace(layer.z_1, layer.z_2, int(nms*100))
            if (layer.material == 'FM' or layer.material == 'HFM'):
                A = self.x[i * 8:i * 8 + 4]
                B = self.x[i * 8 + 4:i * 8 + 8]
                Tx, Ty, Tz = layer.spin_torque(A, B, z)
            else:
                Tx, Ty, Tz = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z))
            T_tot[i,0] = np.sum(Tx)*np.abs(z[1]-z[0])
            T_tot[i,1] = np.sum(Ty)*np.abs(z[1]-z[0])
            T_tot[i,2] = np.sum(Tz)*np.abs(z[1]-z[0])
        for i in range(self.n_interfaces-2):
            if(self.int_parameters[i].type == 'RashbaSpinMixing'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])
                # Torkance contribution
                V_low = inter.lower_layer.charge_potential_and_current(A_low, B_low, z)[0]
                V_high = inter.upper_layer.charge_potential_and_current(A_high, B_high, z)[0]
                mu_low = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                mu_high = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = self.interfaces[i].spin_tensors.torkance_mag[:3,:3]
                Tm = self.interfaces[i].spin_tensors.torkance_mag[:3,3]
                Tx, Ty, Tz = torkance_mag @ (mu_high.flatten() + mu_low.flatten()) + Tm*(V_high - V_low)
                if(self.interfaces[i].full_absorption):
                    # add bulk contributio
                    A = self.interfaces[i].transverse_transformation_tensor()
                    torkance_FM  = self.interfaces[i].spin_tensors.torkance[:3,:3] @ A
                    Tx_FM, Ty_FM, Tz_FM = torkance_FM @ (mu_high.flatten() + mu_low.flatten())
                    Tx += Tx_FM
                    Ty += Ty_FM
                    Tz += Tz_FM

                # torkivity contribution
                torkivity_mag = self.interfaces[i].spin_tensors.torkivity_mag[:3]
                Tx += torkivity_mag[0]*abs(self.E[0])
                Ty += torkivity_mag[1]*abs(self.E[0])
                Tz += torkivity_mag[2]*abs(self.E[0])
                if(self.interfaces[i].full_absorption):
                    A = self.interfaces[i].transverse_transformation_tensor()
                    torkivity_FM  = A @ self.interfaces[i].spin_tensors.torkivity[:3]
                    if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        torkivity_FM*= -1
                    # add bulk contribution
                    Tx += torkivity_FM[0]*abs(self.E[0])
                    Ty += torkivity_FM[1]*abs(self.E[0])
                    Tz += torkivity_FM[2]*abs(self.E[0])

                if(inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    T_tot[i,0] += Tx
                    T_tot[i,1] += Ty
                    T_tot[i,2] += Tz
                else:
                    T_tot[i+1,0] += Tx
                    T_tot[i+1,1] += Ty
                    T_tot[i+1,2] += Tz

            if(self.int_parameters[i].type == 'RashbaPerturbSpinMixing'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])
                # Torkance contribution
                V_low = inter.lower_layer.charge_potential_and_current(A_low, B_low, z)[0]
                V_high = inter.upper_layer.charge_potential_and_current(A_high, B_high, z)[0]
                mu_low = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                mu_high = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = self.interfaces[i].torkance_tensor()[:3,:3]
                Tx, Ty, Tz = torkance_mag @ (mu_high.flatten() + mu_low.flatten())
                Tx *=0
                Ty *=0
                Tz *=0
                if(self.interfaces[i].full_absorption):
                    # add bulk contribution
                    # torkance_FM = self.interfaces[i].MCT_transverse_tensor()[:3,:3] 
                    torkance_FM  = self.interfaces[i].MCT_transverse_transmission_tensor()[:3,:3]
                    Tx_FM, Ty_FM, Tz_FM = torkance_FM @ (mu_high.flatten() + mu_low.flatten())
                    Tx += Tx_FM
                    Ty += Ty_FM
                    Tz += Tz_FM
                # torkivity FM contribution
                if(self.interfaces[i].full_absorption):
                    torkivity_FM_E  = self.interfaces[i].torkivity_tensor()[:3]
                    if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        torkivity_FM_E*= -1
                    # add bulk contribution
                    Tx += torkivity_FM_E[0]
                    Ty += torkivity_FM_E[1]
                    Tz += torkivity_FM_E[2]

                if(inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    T_tot[i,0] += Tx
                    T_tot[i,1] += Ty
                    T_tot[i,2] += Tz
                else:
                    T_tot[i+1,0] += Tx
                    T_tot[i+1,1] += Ty
                    T_tot[i+1,2] += Tz

            if(self.int_parameters[i].type == 'MCT'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])
                if(inter.upper_layer.material == 'FM' or inter.upper_layer.material == 'HFM'):
                    mu_NM = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                else:
                    mu_NM = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                torkance_mag = np.real(self.interfaces[i].MCT_transverse_tensor()[:3,:3])
                Tx, Ty, Tz = torkance_mag @ mu_NM.flatten()

                if (inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    T_tot[i,0] += Tx
                    T_tot[i,1] += Ty
                    T_tot[i,2] += Tz
                else:
                    T_tot[i+1,0] += Tx
                    T_tot[i+1,1] += Ty
                    T_tot[i+1,2] += Tz

            if(self.int_parameters[i].type == 'MCT_Rashba'):
                A_low = self.x[i * 8:i * 8 + 4]
                B_low = self.x[i * 8 + 4:i * 8 + 8]
                A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                inter = self.interfaces[i]
                z  = np.array([self.geometry[i+1]])
                # Torkance contribution
                V_low = inter.lower_layer.charge_potential_and_current(A_low, B_low, z)[0]
                V_high = inter.upper_layer.charge_potential_and_current(A_high, B_high, z)[0]
                mu_low = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                mu_high = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                if(inter.transmission):
                    torkance_mag = self.interfaces[i].torkance_tensor()[:3,:3]
                else:
                    torkance_mag = np.zeros((3,3))
                Tx, Ty, Tz = torkance_mag @ (mu_high.flatten() + mu_low.flatten())
                # Tx *=0
                # Ty *=0
                # Tz *=0
                if(self.interfaces[i].full_absorption):
                    # add bulk contribution
                    # torkance_FM = self.interfaces[i].MCT_transverse_tensor()[:3,:3] 
                    torkance_FM  = self.interfaces[i].MCT_transverse_transmission_tensor()[:3,:3]
                    Tx_FM, Ty_FM, Tz_FM = torkance_FM @ (mu_high.flatten() + mu_low.flatten())
                    Tx += Tx_FM
                    Ty += Ty_FM
                    Tz += Tz_FM
                # torkivity FM contribution
                if(self.interfaces[i].full_absorption):
                    torkivity_FM_E  = self.interfaces[i].torkivity_tensor()[:3]
                    if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                        torkivity_FM_E*= -1
                    # add bulk contribution
                    Tx += torkivity_FM_E[0]
                    Ty += torkivity_FM_E[1]
                    Tz += torkivity_FM_E[2]

                if(inter.lower_layer.material == 'FM' or inter.lower_layer.material == 'HFM'):
                    T_tot[i,0] += Tx
                    T_tot[i,1] += Ty
                    T_tot[i,2] += Tz
                else:
                    T_tot[i+1,0] += Tx
                    T_tot[i+1,1] += Ty
                    T_tot[i+1,2] += Tz

                # if(self.interfaces[i].full_absorption):
                #     A_low = self.x[i * 8:i * 8 + 4]
                #     B_low = self.x[i * 8 + 4:i * 8 + 8]
                #     A_high = self.x[(i+1) * 8:(i+1) * 8 + 4]
                #     B_high = self.x[(i+1) * 8 + 4:(i+1) * 8 + 8]
                #     inter = self.interfaces[i]
                #     z  = np.array([self.geometry[i+1]])
                # # torkance contribution
                # if(inter.upper_layer.material == 'FM'):
                #     mu_NM = inter.lower_layer.spin_accumulation(A_low, B_low, z,False)
                # else:
                #     mu_NM = inter.upper_layer.spin_accumulation(A_high, B_high, z,False)
                # torkance_mag = np.real(self.interfaces[i].MCT_transverse_tensor()[:3,:3])
                # Tx, Ty, Tz = torkance_mag @ mu_NM.flatten()
                # torkivity_FM_E  = self.interfaces[i].torkivity_tensor()[:3]
                # if (self.interfaces[i].lower_layer.material == 'FM' or self.interfaces[i].lower_layer.material == 'HFM'):
                #     torkivity_FM_E*= -1
                # # add bulk contribution
                # Tx += torkivity_FM_E[0]
                # Ty += torkivity_FM_E[1]
                # Tz += torkivity_FM_E[2]

        # Convert from A/m^2 to A/s 
        if scale:
            T_tot *= mub_/ee_
        return T_tot
