import os
import io
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.distance import cdist
from skimage.transform import resize as resize_image
from skimage import measure
from skimage.color import rgb2gray
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from IPython.display import Image, display, clear_output
from matplotlib.collections import LineCollection
from DeepTopOpt.FEA import LinearElasticity # local library

def plot_design(ax,mesh,rho,vol_field,fixed_dofs,load):
    """Plot the given design (rho) on the provided figure axes

    Args:
    ax(object): figure object axes
    mesh(object): finite element mesh object
    rho(nelx x nely float matrix): density matrix
    vol_field(nelx x nely float matrix): volume field
    fixed_dofs(N x 1 int array): fixed degrees-of-freedom
    load(ndof x 1 float array): load vector

    """

    buffer = 2 # buffer around image to present graphical objects which extend outside the design domain
    pix_offset = 0.5-buffer # pixel center offset
    rho = np.pad(rho,buffer)
    # get volume fraction and calculate difference
    volfrac = np.sum(vol_field)/len(mesh.IX)
    volume_violation = np.sum(rho)/len(mesh.IX) - volfrac
    ax.set_title("Volfrac: "+str(round(volfrac,3))+" Vol. diff.: "+str(round(volume_violation,3)))

    # plot design
    ax.imshow(-rho,cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    # plot loads
    load_nodes = np.unique(mesh.dof2nodeid(np.nonzero(load)[0]))
    for node in load_nodes:
        magx = load[2*node][0]
        magy = -load[2*node+1][0] # reverse due to opposite y-coordinate for images and mesh
        xn = mesh.XY[node,0]
        yn = mesh.XY[node,1]
        ax.arrow(xn-pix_offset,yn-pix_offset,magx,magy,width=0.5,color='r')

    # plot boundary conditions
    even_dofs = fixed_dofs%2==0
    uneven_dofs = even_dofs!=True
    horz_bound_nodes = mesh.dof2nodeid(fixed_dofs[even_dofs])
    vert_bound_nodes = mesh.dof2nodeid(fixed_dofs[uneven_dofs])
    ax.scatter(mesh.XY[horz_bound_nodes][:,0]-pix_offset,mesh.XY[horz_bound_nodes][:,1]-pix_offset,marker='>',color='b')
    ax.scatter(mesh.XY[vert_bound_nodes][:,0]-pix_offset,mesh.XY[vert_bound_nodes][:,1]-pix_offset,marker='^',color='g')
    ax.axis('off')

def plot_field(ax,field_data):
    """Plot 2-d field data with a fitted colorbar on the provided figure axes

    Args:
    ax(object): figure object axes
    field_data(nelx x nely float matrix): element-wise field values

    """

    im = ax.imshow(field_data, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)
    cbar = plt.colorbar(im,cax=cax)
    cbar.ax.locator_params(nbins=8)


def train_plot(mesh,compliance,volume_violation,rho,psi,vol_field,fixed_dofs,load):
    """Plots convergence curve along with two random designs from the batch.
    Mostly used for evaluation of training procedure in jupyter notebooks

    Args:
    mesh(object): finite element mesh object
    compliance(Nx1 float array): torch array with compliance values for each train iteration
    volume_violation(Nx1 float array): torch array with volume violation values for each train iteration
    rho(B x 1 x nely x nelx float matrix): torch matrix with densities for each entry in the batch
    fixed_dofs(B x N float matrix): numpy matrix with fixed_dofs for each entry in the batch
    load(B x ndof x 1 float matrix): numpy matrix with loads for each entry in the batch

    """

    # transform from tensors to numpy
    rho = rho.squeeze(1).detach().cpu().numpy()
    psi = psi.cpu().squeeze(1).numpy()
    vol_field = vol_field.cpu().squeeze(1).numpy()
    fixed_dofs = fixed_dofs.numpy()
    load = load.numpy()

    tmp_img = "tmp_design_out.png"

    fig = plt.figure(constrained_layout=True,figsize=(12,6))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    # plot convergence
    ax0.set_xlabel('Iter')
    ax0.set_ylabel('Compliance')
    ax0.plot(np.arange(len(compliance)), compliance,color='b')
    ax0.tick_params(axis='y', labelcolor='b')
    ax0_twin = ax0.twinx()
    ax0_twin.set_ylabel('Vol. violation')
    ax0_twin.plot(np.arange(len(volume_violation)), volume_violation,color='r')
    ax0_twin.tick_params(axis='y', labelcolor='r')

    # plot designs
    plot_design(ax1,mesh,rho[0],vol_field[0],remove_padding(fixed_dofs[0],-1),load[0])
    plot_design(ax2,mesh,rho[1],vol_field[1],remove_padding(fixed_dofs[1],-1),load[1])
    #ax2.imshow(psi[0],cmap="gray")
    #ax2.axis("off")

    plt.savefig(tmp_img)
    plt.close(fig)
    display(Image(filename=tmp_img))
    clear_output(wait=True)
    os.remove(tmp_img)

def tensorboard_plot(mesh,rho,vol_field,fixed_dofs,load):
    """Function used to create images saved by tensorboard"""
    # transform inputs from tensors to numpy
    rho = rho.squeeze(1).detach().cpu().numpy()
    vol_field = vol_field.cpu().squeeze(1).numpy()
    fixed_dofs = fixed_dofs.numpy()
    load = load.numpy()
    # setup plot
    rows = 4
    cols = 2
    fig, axarr = plt.subplots(rows,cols,figsize=(12,12))
    it=0
    for i in range(rows):
        for j in range(cols):
            ax = axarr[i][j]
            plot_design(ax,mesh,rho[it],vol_field[it],remove_padding(fixed_dofs[it],-1),load[it])
            it+=1
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_grad_flow(named_parameters):
    """Plots the gradient flow through each of the layers in the network"""
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def remove_padding(X,pad_value):
    """Convience function used to remove padding from inputs which have been
    padded during batch generation"""
    return X[X!=pad_value]

def count_model_parameters(model):
    """Count trainable parameters of a given model

    Args:
    model(object): torch model object

    Returns:
    trainable_parameters(int): number of trainable parameters

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_data(x):
    """Normalize data to values between 0-1

    Args:
    x(NxN float matrix): data to be normalized

    Returns:
    x_norm((NxN float matrix): normalized data

    """

    return (x-x.min())/(x.max()-x.min())

def standardize_data(x):
    """Standardize data by subtrating mean and dividing with standard deviation

    Args:
    x(NxN float matrix): data to be standardized

    Returns:
    x_norm((NxN float matrix): standardized data

    """

    return (x-x.mean())/x.std()

class StreamlineGenerator():
    """Class used to generate streamlines

    Args:
    nelx(int): number of elements in x-direction of the mesh
    nely(int): number of elements in y-direction of the mesh
    U(nely x nelx float matrix): velocity-direction in x
    V(nely x nelx float matrix): velocity-direction in y
    vmag(nely x nelx float matrix): magnitude of velocity field
    min_length(float): minimum length of streamlines
    color(str): plotting color of streamlines
    """

    def __init__(self,nelx,nely,U,V,vmag,min_length,color):
        self.nelx = nelx
        self.nely = nely
        self.rscale = min(nelx,nely)/10
        self.interpU = RectBivariateSpline(np.arange(nely), np.arange(nelx), U)
        self.interpV = RectBivariateSpline(np.arange(nely), np.arange(nelx), V)
        self.interpMag = RectBivariateSpline(np.arange(nely), np.arange(nelx), vmag)
        self.min_length = min_length
        self.color = color

    def integrate_streamline(self,xpos,ypos,max_iter=100):
        """Euler integration of a streamline from starting point (xpos,ypos)"""
        x_line = [xpos]
        y_line = [ypos]
        v_line = []
        dt = self.rscale
        it = 0
        while (xpos>=0 and xpos<=self.nelx) and (ypos>=0 and ypos<=self.nely):
            # calculate velocity-direction and magnitude in a given point
            u = self.interpU(ypos,xpos)[0][0]
            v = self.interpV(ypos,xpos)[0][0]
            vmag = self.interpMag(ypos,xpos)[0][0]
            # update positions
            xpos+=dt*u
            ypos+=dt*v
            # save points
            x_line.append(xpos)
            y_line.append(ypos)
            v_line.append(vmag)
            # terminate if velocity is too small or maximum number of iterations is reached
            it+=1
            if it>=max_iter or vmag<1e-4:
                break
        return np.array(x_line), np.array(y_line), np.array(v_line)

    def generate_streamlines(self,seed_points):
        """Generate streamlines based on a set of seed points"""
        streamlines = []
        # loop over all seed points
        for idx,(x0,y0) in enumerate(seed_points):
            x_strm,y_strm,v_strm = self.integrate_streamline(x0,y0)
            dx = np.abs(x_strm[-1]-x_strm[0])
            dy = np.abs(y_strm[-1]-y_strm[0])
            line_length = np.sqrt(dx**2+dy**2)
            # save to matplotlib line collection if streamline is longer than the
            # specified minimum length threshold
            if line_length>self.min_length:
                lc = self.create_linecollection(x_strm,y_strm,v_strm)
                streamlines.append(lc)
        return streamlines

    def create_linecollection(self,x,y,v):
        """Creates matplotlib linecollection based on the set of points in the streamline"""
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lw = v*20+2
        lc = LineCollection(segments, linewidths=lw,color=self.color)
        return lc

def stream2grayscale(strm1_lc,strm2_lc,nelx,nely,dpi):
    """Given two line collections of streamlines create a grayscale image

    Args:
    strm1_lc(matplotlib.LineCollection): streamlines corresponding to principal stress direction 1
    strm2_lc(matplotlib.LineCollection): streamlines corresponding to principal stress direction 2

    Returns:
    img_gray(nely*dpi/10 x nelx*dpi/10): grayscale image
    """
    # plot the two line collections
    fig,ax = plt.subplots(1,1,figsize=(12,6),dpi=dpi)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    for lc in strm1_lc:
        ax.add_collection(lc)
    for lc in strm2_lc:
        ax.add_collection(lc)
    ax.set_xlim([0,nelx])
    ax.set_ylim([0,nely])
    ax.invert_yaxis()
    plt.margins(0,0)
    # save the image as a buffer
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=dpi)
    io_buf.seek(0)
    # create numpy array from buffer
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    # save numpy array as grayscale image
    img_gray = rgb2gray(img_arr)
    plt.close()
    return img_gray

def density_sort_threshold(rho,volfrac,Emin=1e-9):
    """Sort densities and threshold based on volume constraint"""
    _,nely,nelx = rho.shape
    rho_flat = rho.flatten()
    vol_idx = int(np.floor(volfrac*nelx*nely))
    ind = np.argsort(rho_flat)[::-1]
    rho_flat[ind[:vol_idx]] = 1
    rho_flat[ind[vol_idx:]] = 1e-9
    rho_thres = rho_flat.reshape((nely,nelx))
    return rho_thres

def remove_disconnects(rho):
    """Use connected components analysis to identify disconnected regions and remove them"""
    # connected components analysis
    label_img, nr_labels = measure.label(rho,background=0,return_num=True)
    # only keep the two largest labels (background + largest component)
    max_labels = np.argsort([np.sum(label_img==i) for i in range(nr_labels+1)])[-2:]
    # mask on all labels not part of the largest components
    small_label_mask = np.logical_and(label_img!=max_labels[0],label_img!=max_labels[1])
    # set all small labels to background
    label_img[small_label_mask] = 0
    # convert to zero-one
    label_img[label_img>0] = 1
    return label_img

def postprocess_designs(rho,vol_field):
    """Post process a batch of designs by first using a threshold based
    on density sorting, and then a connected component analysis"""
    batch_size = rho.shape[0]
    device = rho.device
    # convert input tensors to cpu numpy arrays
    rho = rho.cpu().numpy()
    vol_field = vol_field.cpu().numpy()
    for i in range(batch_size):
        volfrac = vol_field[i,0,0,0]
        rho[i] = density_sort_threshold(rho[i],volfrac)
        rho[i] = remove_disconnects(rho[i])
    # move rho to gpu
    rho = torch.tensor(rho,dtype=torch.float32).to(device)
    return rho

class DataGen:
    """Class for generating training and test data

    Args:
    mesh(object): finite element mesh object
    volfrac_range(2x1 float array): array with lowest and highest volume fraction
    load_range(2x2 float array): array indicating domain where a load may be applied
    n_bc_samples(int): number of samples per boundary condition

    """

    def __init__(self,mesh,volfrac_range,load_range,n_bc_samples):
        self.mesh = mesh
        self.volfrac_range = volfrac_range
        self.load_x_range = load_range[0]
        self.load_y_range = load_range[1]
        self.n_bc_samples = n_bc_samples

    def gen_volfracs(self,n_samples):
        """Generate a specified number of volume fractions within the allowed range"""
        return np.random.uniform(self.volfrac_range[0],self.volfrac_range[1],n_samples)

    def gen_rand_unit_vec(self,ndim):
        """Generate a unit vector with a given dimension"""
        x = np.random.standard_normal(ndim)
        return x / np.linalg.norm(x)

    def gen_rand_unit_vectors(self,ndim,n_samples):
        """Generate a specified number of unit vectors"""
        return [self.gen_rand_unit_vec(ndim) for _ in range(n_samples)]

    def gen_rand_pos(self,x_range,y_range):
        """Generate a random position within the allowed domain"""
        return np.array([np.random.randint(x_range[0],x_range[1]+1),np.random.randint(y_range[0],y_range[1]+1)])

    def gen_rand_positions(self,x_range,y_range,n_samples):
        """Generate a specified number of random positions"""
        return [self.gen_rand_pos(x_range,y_range) for _ in range(n_samples)]

    def gen_load_positions(self,fixed,n_samples):
        """Generate random load positions within the specified load domain
        a new position is generated if the load is too close to the boundary condition """
        bound_rad = max(self.mesh.nelx,self.mesh.nely)//10
        fixed_pos = self.mesh.XY[self.mesh.dof2nodeid(fixed)]
        n_pos = 0
        load_pos_arr = np.empty((0,2))
        while n_pos < n_samples:
            load_pos = np.array(self.gen_rand_positions(self.load_x_range,self.load_y_range,n_samples-n_pos))
            dist_mat = cdist(fixed_pos,load_pos) # get distance between all load positions and boundary positions
            rem_idx = np.unique(np.nonzero(dist_mat<bound_rad)[1]) # remove all indices which violates the boundary radius
            load_pos = np.delete(load_pos,rem_idx,axis=0)
            load_pos_arr = np.append(load_pos_arr, load_pos, axis=0)
            n_pos = len(load_pos_arr)

        return load_pos_arr

    def check_system_conditioning(self,BCs):
        """Check system conditioning by trying to solve the problem with the specified load
        and boundary conditions on a fully solid domain"""
        # insert boundary conditions
        fea = LinearElasticity(self.mesh)
        for bc in BCs:
            if bc[0]=='wall':
                fea.insert_wall_boundary(wall_pos=bc[1],wall_ax=bc[2],bound_dir=bc[3])
            elif bc[0]=='point':
                fea.insert_point_boundary(bound_pos=bc[1],bound_dir=bc[2])
        # insert 45 deg point load in the middle of domain
        fea.insert_point_load(load_pos=[self.mesh.nelx//2,self.mesh.nely//2],load_mag=[1,1])
        # check conditioning of system
        try:
            rho = np.ones((self.mesh.nelx,self.mesh.nely))
            U = fea.solve_system(rho,sparse=True)
        except:
            print("BC:",bc)
            print("Unconditioned system matrix")

    def generate_dataset_dicts(self,bc_cmbs,testset_indices,trainset=True,cond_check=True):
        """Generate a dictionary containing the volume fraction, load and boundary conditions
        for each sample in the dataset

        Args:
        bc_cmbs(list): A list of boundary condition combinations, an example would be
        [["wall",0,"y","xy"],"point",(30,10),"xy"],[...]]
        testset_indices(integer list): indices in the bc_cmbs list which belong to the testset
        trainset(bool): boolean specifying whether to generate train or testset

        Returns:
        dataset_dicts(list): list of dictionaries containing information about each sample
        in the dataset
        """
        # run initial test to check system conditioning
        if cond_check==True:
            print("Running system conditioning test")
            for cmb in bc_cmbs:
                self.check_system_conditioning(cmb)
            print("System conditioning test parsed")
        # loop over all bc combinations and generate data samples
        dataset_dicts = []
        for idx,cmb in enumerate(bc_cmbs):
            fea = LinearElasticity(self.mesh)
            # skip combinations based on whether training or test set is being generated
            if trainset==True:
                if idx in testset_indices:
                    continue
            else:
                if idx not in testset_indices:
                    continue
            # insert boundary conditions(s) in physics class
            wall_bc_list = []
            point_bc_list = []
            for bc in cmb:
                if bc[0]=='wall':
                    fea.insert_wall_boundary(wall_pos=bc[1],wall_ax=bc[2],bound_dir=bc[3])
                    wall_bc_list.append(bc)
                elif bc[0]=='point':
                    fea.insert_point_boundary(bound_pos=bc[1],bound_dir=bc[2])
                    point_bc_list.append(bc)
            # generate volume fractions and load positioning and magnitude
            volfracs = self.gen_volfracs(self.n_bc_samples)
            load_magnitudes = self.gen_rand_unit_vectors(2,self.n_bc_samples)
            load_positions = self.gen_load_positions(fea.fixed_dofs,self.n_bc_samples)
            # save sample parameters in list of dictionaries
            for (load_pos,load_mag,vol) in zip(load_positions,load_magnitudes,volfracs):
                dataset_dicts.append({"wall_BC":wall_bc_list,"point_BC":point_bc_list,
                                     "load_pos":load_pos,"load_mag":load_mag,"volfrac":vol})

        return dataset_dicts
