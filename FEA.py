from __future__ import division # ensures floating point division
import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cholesky

class FEmesh:
    """Class for storing the attributes of a rectangular finite element mesh

    Attributes:
    self.nelx(int): number of elements in x-direction
    self.nely(int): number of elements in y-direction
    self.nele(int): number of elements in mesh
    self.nnodes(int): number of nodes in mesh
    self.ndof(int): number of degrees-of-freedom in mesh
    self.XY(nnodes x 2 int matrix): mesh node coordinates
    self.IX(nele x 4 int matrix): element connectivity matrix
    self.edofMat(nele x 8 int matrix): matrix containing degrees-of-freedom for each element

    Mesh conventions:
    - Node numbering - column-wise, starting at the top
    0---3---6
    |   |   |
    1---4---7
    |   |   |
    2---5---8
    - Element coordinate system
    (-1,1)----(1,1)
    |             |
    |    (0,0)    |
    |             |
    (-1,-1)--(1,-1)
    - Face counts
    *---3---*
    |       |
    4       2
    |       |
    *---1---*
    """

    def __init__(self,nelx,nely):
        self.nelx = nelx
        self.nely = nely
        self.nele = nelx*nely
        self.nnodes = (nelx+1)*(nely+1)
        self.ndof = 2*self.nnodes
        Yv,Xv = np.meshgrid(range(nely+1),range(nelx+1))
        self.XY = np.vstack((Xv.flatten(),Yv.flatten())).T
        self.IX = self.get_connectivity_matrix()
        self.edofMat = self.get_edof_matrix()

    def get_connectivity_matrix(self):
        """Function for creating the connectivity matrix between elements and
        nodes in the mesh

        Returns:
        IX(nele x 4 int matrix): element connectivity matrix

        """
        elx,ely = np.meshgrid(np.arange(self.nelx),np.arange(self.nely))
        elx_vec = elx.flatten(order='F')
        ely_vec = ely.flatten(order='F')
        node0 = (self.nely+1)*elx_vec+(ely_vec+1)
        node1 = (self.nely+1)*(elx_vec+1)+(ely_vec+1)
        node2 = (self.nely+1)*(elx_vec+1)+(ely_vec)
        node3 = (self.nely+1)*elx_vec+ely_vec
        IX = np.vstack((node0,node1,node2,node3)).T
        
        return IX

    def get_edof_matrix(self):
        """Function for creating the element degrees-of-freedom matrix which
        holds the degrees-of-freedom for each element in the mesh

        Returns:
        edofMat(nele x 8 int matrix): matrix containing degrees-of-freedom for each element

        """
        edofMat = np.zeros((self.nele,8),dtype=int)
        edofMat[:,0:8:2] = self.IX*2
        edofMat[:,1:8:2] = self.IX*2+1

        return edofMat

    def eval_shape_func(self,xi,eta):
        """Function for evaluation the rectangular shape function at a given point xi,eta

        Args:
        xi(float): element evaluation point in x
        eta(float): element evaluation point in y

        Returns:
        N (1x4 float array): shape function value at the four element nodes
        dNxi (1x4 float array): differentiated shape function value at the four element nodes with respect to xi
        dNeta (1x4 float array): differentiated shape function value at the four element nodes with respect to eta
        dNmat (4x8 float matrix): differentiated shape function matrix

        """
        # Shape function values
        N = 1/4*np.array([(1-eta)*(1-xi),(1-eta)*(1+xi),(1+eta)*(1+xi),(1-xi)*(1+eta)])
        # Differentiated shape function values
        dNxi = 1/4*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dNeta = 1/4*np.array([-(1-xi), -(1+xi), (1+xi),  (1-xi)])

        # Shape function matrix
        Nmat = np.array([[N[0], 0, N[1], 0, N[2], 0, N[3], 0],
                        [0, N[0], 0, N[1], 0, N[2], 0, N[3]]])

        dNmat = np.zeros((4,8))
        dNmat[0,0:8:2] = dNxi
        dNmat[1,0:8:2] = dNeta
        dNmat[2,1:8:2] = dNxi
        dNmat[3,1:8:2] = dNeta

        return dNxi,dNeta,dNmat

    def dof2nodeid(self,dof):
        """Function for getting the node associated with a given degree-of-freedom

        Args:
        dof(int): degree-of-freedom

        Returns:
        node(int): node index

        """
        return np.floor(dof/2).astype(int)


class LinearElasticity():
    """Class for storing functions and attributes related to solving linear elasticity problems

    Attributes:
    self.mesh(object): finite element mesh object
    self.poisson_ratio(float): poisson ratio of the desired material
    self.youngs_modulus(float): youngs modulus of the desired material
    self.Emin(float): minimum value of youngs modulus, used for computational stability
    self.Emax(float): maximum value of youngs modulus, used for computational stability
    self.penal(float): stiffness penalty parameter for intermediate material densities
    self.fixed_dofs(Nx1 int array): fixed degrees-of-freedom
    self.load(ndofx1 int array): load vector
    self.stiffness_matrix(ndof x ndof float matrix): stiffness matrix
    self.displacement(ndof x 1 float array): displacement vector

    """
    def __init__(self,mesh):
        self.mesh = mesh
        self.poisson_ratio = 0.3
        self.youngs_modulus = 1
        self.Emin = 1e-9*self.youngs_modulus
        self.Emax = self.youngs_modulus
        self.penal = 1
        self.fixed_dofs = []
        self.load = np.zeros((mesh.ndof,1))
        self.Cmat = self.constitutive_stress_matrix()
        self.B0 = self.strain_displacement_matrix()
        self.KE = self.element_stiffness_matrix()

    def element_stiffness_matrix(self):
        """Function for creating the analytical element stiffness matrix

        Returns:
        KE(8x8 float matrix): element stiffness matrix

        """
        E = self.youngs_modulus
        nu = self.poisson_ratio
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])

        return (KE)

    def preIntegrated_force_vectors(self):
        """Function for creating the pre-integrated force vectors for loading of
        entire elements or element faces

        Local node numbering (different from global)
        1-------2
        |       |
        |       |
        4-------3

        Returns:
        ele_force_vecs(tuple): tuple of element forces, each a 1x8 float array

        """
        # unit dimensions for element
        a = 0.5
        b = 0.5
        # initialize traction to unity
        tr11 = 1
        tr22 = 1
        # initialize body force to unity
        bf11 = 1
        bf22 = 1

        f0x_face1 = np.array([0, 0, 0, 0, tr11 * a, 0, tr11 * a, 0,])
        f0y_face1 = np.array([0, 0, 0, 0, 0, tr22 * a, 0, tr22 * a,])
        f0x_face2 = np.array([0, 0, tr11 * b, 0, tr11 * b, 0, 0, 0,])
        f0y_face2 = np.array([0, 0, 0, tr22 * b, 0, tr22 * b, 0, 0,])
        f0x_face3 = np.array([tr11 * a, 0, tr11 * a, 0, 0, 0, 0, 0,])
        f0y_face3 = np.array([0, tr22 * a, 0, tr22 * a, 0, 0, 0, 0,])
        f0x_face4 = np.array([tr11 * b, 0, 0, 0, 0, 0, tr11 * b, 0,])
        f0y_face4 = np.array([0, tr22 * b, 0, 0, 0, 0, 0, tr22 * b,])
        f0x_element = np.array([a * b * bf11, 0, a * b * bf11, 0, a * b * bf11, 0, a * b * bf11, 0,])
        f0y_element = np.array([0, a * b * bf22, 0, a * b * bf22, 0, a * b * bf22, 0, a * b * bf22,])
        ele_force_vecs = (f0x_face1,f0y_face1,f0x_face2,f0y_face2,f0x_face3,f0y_face3,f0x_face4,f0y_face4,f0x_element,f0y_element)
        return ele_force_vecs

    def strain_displacement_matrix(self):
        """Function for creating strain-displacement matrix based on shape functions

        Returns:
        B0(3x8 float matrix): strain-displacement matrix
        """
        L = np.zeros((3,4)) # matrix used for mapping shape function derivatives into strains
        L[0,0] = 1; L[1,3] = 1; L[2,1:3] = 1;
        xi = 0; eta = 0; # evaluation point
        _,_,dNmat = self.mesh.eval_shape_func(xi,eta) # differentiated shape functions evaluated in (xi,eta)
        Jac = np.array([[0.5,0],[0,0.5]]) # mapping back to global coordinates
        detJ = np.linalg.det(Jac)
        invJ = 1/detJ*np.array([[Jac[1,1], -Jac[0,1]], [-Jac[1,0], Jac[0,0]]])
        G = np.zeros((4,4)) # mapping matrix between rectangular element space coordinates and global coordinates
        G[:2,:2] = invJ; G[-2:,-2:] = invJ
        B0 = np.matmul(np.matmul(L,G),dNmat) # strain displacement matrix

        return B0

    def constitutive_stress_matrix(self):
        """Function for creating the constitutive stiffness matrix

        Returns:
        Cmat(3x3 float matrix): constitutive stiffness matrix
        """
        E = self.youngs_modulus
        nu = self.poisson_ratio
        Cmat = E/(1-nu**2)*np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]])

        return Cmat

    def stiffness_matrix_assembly(self,rho,sparse=True):
        """Assembles the global stiffness from the local element matrices

        Args:
        rho(nely x nelx float matrix): density matrix
        sparse(bool): matrix type

        Returns:
        stiffness_matrix(ndof x ndof float matrix): global stiffness matrix

        """
        rho = rho.flatten(order='F')
        if sparse is False:
            self.stiffness_matrix = np.zeros((self.mesh.ndof,self.mesh.ndof))
            mat_idx = np.indices((8,8))
            I = mat_idx[0].flatten()
            J = mat_idx[1].flatten()
            # assemble stiffness matrix
            for ele in range(len(self.mesh.edofMat)):
                edof = self.mesh.edofMat[ele]
                self.stiffness_matrix[edof[I],edof[J]] += (self.Emin+(rho[ele])**self.penal*(self.Emax-self.Emin))*self.KE[I,J]
        elif sparse is True:
            iK = np.kron(self.mesh.edofMat,np.ones((8,1))).flatten()
            jK = np.kron(self.mesh.edofMat,np.ones((1,8))).flatten()
            sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(rho)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
            self.stiffness_matrix = sp.coo_matrix((sK,(iK,jK)),shape=(self.mesh.ndof,self.mesh.ndof)).tocsc()

        return self.stiffness_matrix

    def solve_system(self,rho,unit_load=False,sparse=True):
        """Solves the linear system Ku=F

        Args:
        rho(nely x nelx float matrix): density matrix
        unit_load(bool): normalize load to unit magnitude
        sparse(bool): matrix type

        Returns:
        U(ndof x 1 float array): displacement vector

        """

        # check whether load and boundary conditions have been applied
        if len(self.fixed_dofs)==0 or len(np.nonzero(self.load)[0])==0:
            raise AttributeError("Please insert load and boundary conditions before solving the system")

        # normalize load if flag is true
        if unit_load is True:
            load_sum = np.sum(np.abs(self.load))
            self.load/=load_sum

        # assemble the stiffness matrix
        if sparse is False:
            self.stiffness_matrix_assembly(rho,sparse=False)
            # apply BC using zero-one method
            self.stiffness_matrix[self.fixed_dofs,:] = 0
            self.stiffness_matrix[:,self.fixed_dofs] = 0
            self.stiffness_matrix[self.fixed_dofs,self.fixed_dofs] = 1
            self.load[self.fixed_dofs] = 0
            # solve system
            U = np.linalg.solve(self.stiffness_matrix,self.load)
            self.displacement = U
        elif sparse is True:
            self.stiffness_matrix_assembly(rho,sparse=True)
            # get free dofs
            dofs=np.arange(2*(self.mesh.nelx+1)*(self.mesh.nely+1))
            free_dofs=np.setdiff1d(dofs,self.fixed_dofs)
            # only solve system for free dofs
            U=np.zeros((self.mesh.ndof,1))
            Kfree = self.stiffness_matrix[free_dofs,:][:,free_dofs]
            factor = cholesky(Kfree)
            U[free_dofs,0] = factor(self.load[free_dofs,0])
            self.displacement = U.squeeze(-1)

        return self.displacement

    def compute_compliance(self,sparse=True):
        """Calculates the compliance of an already solved system with a stored displacement vector

        Returns:
        compliance(float): compliance value (U^T*K*U)

        """
        if sp.issparse(self.stiffness_matrix)==True:
            compliance = self.stiffness_matrix.dot(self.displacement).dot(self.displacement)
        else:
            compliance = np.matmul(np.matmul(self.displacement.T,self.stiffness_matrix),self.displacement)
        return compliance.item()

    def insert_wall_boundary(self,wall_pos,wall_ax,bound_dir):
        """Modifies the fixed_dofs vector to incorporate a new wall boundary condition

        Args:
        wall_pos(float): wall position in the mesh
        wall_ax(float): mesh axis along which the boundary condition is enforced (options: "x" or "y")
        bound_dir(float): direction along which the boundary condition is enforced (options: "x","y" or "xy")

        """

        if wall_ax == "y":
            wall_nodes = np.nonzero(self.mesh.XY[:,0]==wall_pos)[0]
        elif wall_ax == "x":
            wall_nodes = np.nonzero(self.mesh.XY[:,1]==wall_pos)[0]
        else:
            print("Invalid wall direction value")

        if bound_dir=='x':
            bound_dof = wall_nodes*2
        elif bound_dir=='y':
            bound_dof = wall_nodes*2+1
        elif bound_dir=='xy':
            bound_dof = np.union1d(wall_nodes*2,wall_nodes*2+1)
        else:
            print("Invalid boundary direction value")

        self.fixed_dofs = np.union1d(self.fixed_dofs,bound_dof).astype(int)

    def insert_point_boundary(self,bound_pos,bound_dir):
        """Modifies the fixed_dofs vector to incorporate a new point boundary condition

        Args:
        bound_pos(2x1 float array): boundary position in the mesh
        bound_dir(float): direction along which the boundary condition is enforced (options: "x","y" or "xy")

        """

        xb = bound_pos[0]; yb = bound_pos[1]
        bound_node = np.nonzero(np.logical_and(self.mesh.XY[:,0]==xb,self.mesh.XY[:,1]==yb))[0]
        if bound_dir=='x':
            bound_dof = np.array([bound_node*2])
        elif bound_dir=='y':
            bound_dof = np.array([bound_node*2+1])
        elif bound_dir=='xy':
            bound_dof = np.array([bound_node*2,bound_node*2+1])
        else:
            print("Invalid boundary direction value")

        self.fixed_dofs = np.union1d(self.fixed_dofs,bound_dof).astype(int)

    def insert_point_load(self,load_pos,load_mag):
        """Modifies the load vector to incorporate a new point load

        Args:
        load_pos(2x1 float array): boundary position in the mesh
        load_mag(2x1 float array): load magnitude in x and y direction

        """

        xf = load_pos[0]; yf = load_pos[1]
        load_node = np.nonzero(np.logical_and(self.mesh.XY[:,0]==xf,self.mesh.XY[:,1]==yf))[0]
        self.load[load_node*2] = load_mag[0]
        self.load[load_node*2+1] = load_mag[1]

    def insert_face_forces(self,ele_ids,ele_faces,face_mags):
        """
        Args:
        ele_ids(integer array): array of element ids for which a face force is applied
        ele_faces(integer array): array indicating which element face to apply the force
        face_mags(2xN float array): array indicating the magnitude of the force in each direction

        """
        f0x_face1,f0y_face1,f0x_face2,f0y_face2,f0x_face3,f0y_face3,f0x_face4,f0y_face4,_,_ = self.preIntegrated_force_vectors()
        for i,id in enumerate(ele_ids):
            edof = self.mesh.edofMat[id].squeeze()
            eface = ele_faces[i]
            if eface==1:
                fvec = f0x_face1*face_mags[eface,0] + f0y_face1*face_mags[eface,1]
            elif eface==2:
                fvec = f0x_face2*face_mags[eface,0] + f0y_face2*face_mags[eface,1]
            elif eface==3:
                fvec = f0x_face3*face_mags[eface,0] + f0y_face3*face_mags[eface,1]
            elif eface==4:
                fvec = f0x_face4*face_mags[eface,0] + f0y_face4*face_mags[eface,1]
            else:
                raise ValueError("Quad-element faces should be in the range 1-4")
            self.load[edof] += fvec[:,np.newaxis]

    def insert_element_forces(self,ele_ids,ele_mags):
        """
        Args:
        ele_ids(integer array): array of element ids for which a face force is applied
        ele_mags(2xN float array): array indicating the magnitude of the force in each direction
        """

        _,_,_,_,_,_,_,_,f0x_element,f0y_element = self.preIntegrated_force_vectors()
        for i,id in enumerate(ele_ids):
            edof = self.mesh.edofMat[id].squeeze()
            fvec = f0x_element*ele_mags[i,0] + f0y_element*ele_mags[i,1]
            self.load[edof] += fvec[:,np.newaxis]

    def compute_element_strains(self):
        """Compute the strain-tensor in each element

        Returns:
        tuple(3 x (nely x nelx) float matrices): epsilon_11, epsilon_22, epsilon_12
        """

        epsilon_mat = np.matmul(self.B0,self.displacement[self.mesh.edofMat].T)

        # reshape element-wise strains to matrix
        epsilon_11 = epsilon_mat[0].reshape((self.mesh.nely,self.mesh.nelx),order='F')
        epsilon_22 = epsilon_mat[1].reshape((self.mesh.nely,self.mesh.nelx),order='F')
        # divide by 2 since epsilon vector is defined as [eps_11,eps_22,2*eps_12]
        epsilon_12 = epsilon_mat[2].reshape((self.mesh.nely,self.mesh.nelx),order='F')/2

        return (epsilon_11,epsilon_22,epsilon_12)

    def compute_element_stresses(self):
        """Compute the stress-tensor in each element

        Returns:
        tuple(3 x (nely x nelx) float matrices): sigma_11, sigma_22, sigma_12
        """

        sigma_mat = np.matmul(self.Cmat,np.matmul(self.B0,self.displacement[self.mesh.edofMat].T))

        # reshape element-wise stresses to matrix
        sigma_11 = sigma_mat[0].reshape((self.mesh.nely,self.mesh.nelx),order='F')
        sigma_22 = sigma_mat[1].reshape((self.mesh.nely,self.mesh.nelx),order='F')
        sigma_12 = sigma_mat[2].reshape((self.mesh.nely,self.mesh.nelx),order='F')

        return (sigma_11,sigma_22,sigma_12)

    def compute_strain_energy(self,strain_tensor,stress_tensor):
        """Compute the strain-energy in each element

        Args:
        tuple(3 x (nely x nelx) float matrices): strain tensor
        tuple(3 x (nely x nelx) float matrices): stress tensor

        Returns:
        W(nely x nelx float matrix): strain-density in each element
        """

        eps_11,eps_22,eps_12 = strain_tensor
        sigma_11,sigma_22,sigma_12 = stress_tensor
        W = 1/2*(sigma_11*eps_11+sigma_22*eps_22+2*sigma_12*eps_12)
        return W

    def compute_von_mises_stresses(self,stress_tensor):
        """Compute the von mises stress in each element

        Args:
        tuple(3 x (nely x nelx) float matrices): stress tensor

        Returns:
        sigma_v(nely x nelx float matrix): von mises stress in each element
        """

        sigma_11,sigma_22,sigma_12 = stress_tensor
        return np.sqrt(sigma_11**2+sigma_22**2-sigma_11*sigma_22+3*sigma_12**2)

    def compute_principal_stresses(self,stress_tensor):
        """Compute the principal stresses and direction in each element

        Args:
        tuple(3 x (nely x nelx) float matrices): stress tensor

        Returns:
        sigma_1(nely x nelx float matrix): first principal stress
        sigma_2(nely x nelx float matrix): second principal stress
        psi_1(nely x nelx float matrix): first principal stress direction
        psi_2(nely x nelx float matrix): second principal stress direction
        """

        sigma_11,sigma_22,sigma_12 = stress_tensor
        sigma1 = 1/2*(sigma_11+sigma_22)+np.sqrt(((sigma_11-sigma_22)/2)**2+sigma_12**2)
        sigma2 = 1/2*(sigma_11+sigma_22)-np.sqrt(((sigma_11-sigma_22)/2)**2+sigma_12**2)
        psi1 = np.arctan2(-2*sigma_12,sigma_11-sigma_22)/2
        psi2 = psi1-np.pi/2
        return sigma1,sigma2, psi1, psi2

def restriction_matrix(nelx_f,nely_f,reduction_factor=2):
    """Restriction/prolongation matrix used to map between different levels in the multi-grid methods

    Args:
    nelx_f(int): fine-grid number of elements in x-direction
    nely_f(int): fine-grid number of elements in y-direction
    reduction_factor(int): relative reduction of mesh size

    Returns:
    P(ndof_f x ndof_c sparse float matrix): restriction matrix

    """
    nelx_c = nelx_f//reduction_factor
    nely_c = nely_f//reduction_factor
    ndof_f = 2*(nelx_f+1)*(nely_f+1)
    ndof_c = 2*(nelx_c+1)*(nely_c+1)
    maxnum = nelx_c*nely_c*20 # maximum amount of nnz entries in the prolongation operator
    iP = np.zeros(maxnum).astype(int)
    jP = np.zeros(maxnum).astype(int)
    sP = np.zeros(maxnum)
    weights = np.array([1,0.5,0.25]) # interpolation weights

    cc = 0 # nnz counter
    for nx in range(nelx_c+1):
        for ny in range(nely_c+1):
            col = nx*(nely_c+1)+ny
            # coordinate on fine grid
            nx_f = nx*2
            ny_f = ny*2
            # get node neighbourhood
            x_st_idx = max(nx_f-1,0)
            x_end_idx = min(nx_f+1,nelx_f)
            y_st_idx = max(ny_f-1,0)
            y_end_idx = min(ny_f+1,nely_f)
            for k in range(x_st_idx,x_end_idx+1): # +1 since it should be inclusive
                for l in range(y_st_idx,y_end_idx+1):
                    row = k*(nely_f+1)+l
                    ind = int((nx_f-k)**2+(ny_f-l)**2)
                    cc+=1
                    iP[cc] = 2*row; jP[cc] = 2*col; sP[cc] = weights[ind]
                    cc+=1
                    iP[cc] = 2*row+1; jP[cc] = 2*col+1; sP[cc] = weights[ind]

    P = sp.coo_matrix((sP[:cc+1],(iP[:cc+1],jP[:cc+1])),shape=(ndof_f,ndof_c)).tocsr()

    return P

class SparseMGCG:
    """Sparse multi-grid conjugate gradients method implemented on the CPU

    Attributes:
    self.nl(int): number of levels in the V-cycle
    self.omega(float): jacobi smoothing factor
    self.njac(int): number of jacobi smoothing operations in each level
    self.PL(list): list of restriction matrices mapping from one level to the next

    """

    def __init__(self,nelx,nely,n_levels,njac=2):
        self.nl = n_levels
        self.omega = 0.8
        self.njac = njac
        ndof_c = 2*(nelx//(2**n_levels)+1)*(nely//(2**n_levels)+1)
        if ndof_c<1e3:
            raise RuntimeError("System size at coarse level too small. Please decrease number of levels or use a direct solver")
        else:
            print("Preparing restriction matrices")
            self.PL = []
            for l in range(self.nl):
                P = restriction_matrix(nelx//(2**l),nely//(2**l))
                self.PL.append(P.tocsr())

    def jacobi_smooth(self,x,A,b):
        """Function for performing n jacobi smoothing operations

        Args:
        x(ndof float vector): current solution vector
        A(ndof x ndof sparse float matrix): system matrix
        b(ndof float vector): right-hand side of system

        Returns:
        x(ndof vector): new solution vector
        """

        Adiag = A.diagonal()
        D = sp.diags(Adiag)
        invD = sp.diags(1/Adiag)
        LU = (A - D)
        for _ in range(self.njac):
            x = self.omega*invD.dot(b-LU.dot(x)) + (1-self.omega)*x
        return x

    def Vcycle(self,AL,factor,r,level):
        """ Function for performing one V-cycle iteration

        Args:
        AL(list): system matrix at each level
        factor(cholesky factorization object): pre-computed cholesky factorization at lowest level
        r(ndof float vector): conjugate gradient residual vector
        level(int): current level

        """
        z = r*0 # initialize solution
        z = self.jacobi_smooth(z,AL[level],r) # smooth
        Az = AL[level].dot(z) # restrict
        d = r - Az # new residual
        dh2 = self.PL[level].T.dot(d) # restrict residual
        if level+1==self.nl: # if at last level solve system using direct solver
            vh2 = factor(dh2)
        else: # else move to next level
            vh2 = self.Vcycle(AL,factor,dh2,level+1)
        v = self.PL[level].dot(vh2) # prolong
        z += v
        z = self.jacobi_smooth(z,AL[level],r) # smooth
        return z

    def solve(self,A,b,max_iter=100,rtol=1e-6,conv_criterion='comp',verbose=True):
        """Solve sparse system a*x=b using multi-grid preconditioned conjugate gradient method
        with compliance convergence criterion

        Args:
        A(sparse matrix): sparse system matrix
        b(ndof float vector): right-hand side of system
        max_iter(int): max number of iterations for the MGCG solver
        rtol(float): relative error on the convergence criterion
        conv_criterion(str): whether to use compliance 'comp' or displacement 'disp' as the convergence criterion
        verbose(bool): print relevant output

        Returns:
        u(ndof float vector): solution vector
        """
        
        # precompute system matrix for each level
        if verbose is True: print("Precomputing system matrices at each level...")
        AL = [A]
        for l in range(self.nl):
            AL.append(self.PL[l].T.dot(AL[l]).dot(self.PL[l]))

        # cholesky factorize lowest level matrix
        factor = cholesky(AL[-1])
        u = np.zeros(AL[0].shape[0]) # initialize solution vector as 0
        r = b - AL[0].dot(u)
        if conv_criterion=='comp':
            conv_crit = 1e6
            prev_comp = 1e6
        elif conv_criterion=='disp':
            conv_crit = 1e6
            res0 = np.linalg.norm(b)
        it = 0
        while conv_crit>rtol and it<max_iter:
            z = self.Vcycle(AL,factor,r,level=0)
            rho = r.T.dot(z)
            if it==0:
                p = z
            else:
                beta = rho/rho_p
                p = beta*p + z
            q = AL[0].dot(p)
            dpr = p.T.dot(q)
            alpha = rho/dpr
            u+=alpha*p
            r-=alpha*q
            rho_p = rho
            
            it+=1
            
            if conv_criterion=='comp':
                comp = AL[0].dot(u).dot(u).item()
                conv_crit = np.abs(comp-prev_comp)/prev_comp
                prev_comp = comp
            elif conv_criterion=='disp':
                conv_crit = np.linalg.norm(r)/res0

            if verbose is True:
                print("It. {} Rel. error {} ".format(*[it,conv_crit]))

        return u
    






































