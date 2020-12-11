
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage

# BASICS -----------------------------------------------------------------------

# normalize a numpy array wrt its euclidean norm
def normalize(x):
    return np.divide(x,np.linalg.norm(x,ord=2))

# PLOTTING IN 2D ---------------------------------------------------------------
def plot_2D_evolution(H,traj,**kwargs):
    if isinstance(traj[0],np.ndarray):
        x_track = [v[0] for v in traj]
        y_track = [v[1] for v in traj]
    else:
        x_track = [v.pos[0] for v in traj]
        y_track = [v.pos[1] for v in traj]
    
    # visualize our potential
    padding = 0.1
    x = np.arange(min(x_track)-padding, max(x_track)+padding, 0.1)
    y = np.arange(min(y_track)-padding, max(y_track)+padding, 0.1)
    
    F = np.full((y.size,x.size),np.nan)
    G = np.full((y.size,x.size,2),np.nan)
    for ii in range(y.size):
        for jj in range(x.size):
            v = np.array([x[jj],y[ii]])
            F[ii,jj] = H.eval(v)
            G[ii,jj,:] = H.gradient(v)
            
    xx,yy = np.meshgrid(x,y)
    h = plt.contourf(xx,yy,F,alpha=0.5,cmap='gray')
    plt.title('f')
    plt.plot(x_track,y_track,'b')
    plt.plot(x_track[0],y_track[0],'r',marker="x", markersize=10,label='init')
    plt.legend()
    plt.show()
    
    # default behavior is to plot gradient quiverplot
    if 'gradient' in kwargs and not kwargs['gradient']:
        return
    else:
        fig, ax = plt.subplots(figsize=(15,15))
        h = plt.contourf(xx,yy,F,alpha=0.2)
        q = ax.quiver(xx, yy, G[:,:,0], G[:,:,1])
        plt.title('|grad|')
        plt.show()
    return

def plot_potential(H,lims,**kwargs):
    # visualize our potential
    padding = 0.1
    x = np.linspace(lims[0]-padding, lims[1]+padding, 20)
    y = np.linspace(lims[0]-padding, lims[1]+padding, 20)
    
    F = np.full((y.size,x.size),np.nan)
    G = np.full((y.size,x.size,2),np.nan)
    for ii in range(y.size):
        for jj in range(x.size):
            v = np.array([x[jj],y[ii]])
            F[ii,jj] = H.eval(v)
            G[ii,jj,:] = H.gradient(v)
            
    xx,yy = np.meshgrid(x,y)
    h = plt.contourf(xx,yy,F)
    plt.title('f')
    plt.colorbar()
    plt.show()
    
    xx,yy = np.meshgrid(x,y)
    h = plt.contourf(xx,yy,np.log(F))
    plt.title('log(f)')
    plt.colorbar()
    plt.show()
    
    # default behavior is to plot gradient quiverplot
    if 'gradient' in kwargs and not kwargs['gradient']:
        return
    else:
        fig, ax = plt.subplots(figsize=(15,15))
        h = plt.contourf(xx,yy,F,alpha=0.2)
        q = ax.quiver(xx, yy, G[:,:,0], G[:,:,1])
        plt.title('|grad|')
        plt.show()

# PLOTTING IN 3D ---------------------------------------------------------------

def plot_3D_evolution(H,traj,**kwargs):
    if isinstance(traj[0],np.ndarray):
        x_track = [v[0] for v in traj]
        y_track = [v[1] for v in traj]
        z_track = [v[2] for v in traj]
    else:
        x_track = [v.pos[0] for v in traj]
        y_track = [v.pos[1] for v in traj]
        z_track = [v.pos[2] for v in traj]


    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    step_num = np.divide(list(range(len(traj))),len(traj))
    
    ax.scatter(x_track,y_track, z_track,c=step_num)
    ax.scatter(x_track[-1],y_track[-1], z_track[-1],c='r',marker='x',
        s=150,label='final pt')
    plt.legend()
    plt.show()


    return

# ASSESSING PERFORMANCE --------------------------------------------------------
def mean_convergence(H,traj,**kwargs):
    # a (no. observations) x (dimensions) matrix
    # each "observation" is the position at a time point
    data = np.asarray([v.pos for v in traj])
    H_vals = np.asarray([H.eval(v.pos) for v in traj])
    tsteps = [v.t for v in traj]
    pos_norms = np.linalg.norm(data, ord=2, axis=1)

    sliding_window = 100

    pos_mu = [np.mean(data[(up_to_idx-sliding_window):up_to_idx,:], 
        axis=0) for up_to_idx in range(sliding_window,len(traj))]
    
    norm_mu = [np.mean(pos_norms[(up_to_idx-sliding_window):up_to_idx]) 
         for up_to_idx in range(sliding_window,len(traj))]


    H_mu = [np.mean(H_vals[(up_to_idx-sliding_window):up_to_idx]) 
         for up_to_idx in range(sliding_window,len(traj))]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Convergence of evolutionary processs')
    ax1.plot(tsteps[sliding_window:], norm_mu)
    ax1.set_title('L_2 of mean position')

    ax2.plot(tsteps[sliding_window:], H_mu)
    ax2.set_title('Mean function value')
    
    plt.show()


def Gaussian_convergence(H,traj,**kwargs):
    # a (no. observations) x (dimensions) matrix
    # each "observation" is the position at a time point
    data = np.asarray([v.pos for v in traj])
    tsteps = [v.t for v in traj]

    true_mu = H.mu
    true_sigma = H.cov

    sliding_window = 100
    est_mu = [np.mean(data[(up_to_idx-sliding_window):up_to_idx,:], 
        axis=0) for up_to_idx in range(sliding_window,len(traj))]
    est_sigma = [np.cov(data[(up_to_idx-sliding_window):up_to_idx,:],
        rowvar=0) for up_to_idx in range(sliding_window,len(traj))]

    mu_L2_error = [np.linalg.norm(true_mu-est) for est in est_mu]
    cov_F_error =[np.linalg.norm(true_sigma-est,ord='fro') for est in est_sigma]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Convergence to Gaussian parameters')
    ax1.plot(tsteps[sliding_window:], mu_L2_error)
    ax1.set_title('L2 error in mean')

    ax2.plot(tsteps[sliding_window:], 
        np.divide(cov_F_error,np.linalg.norm(true_sigma,ord=2)))
    ax2.set_title('Relative Frobenius error in covariance')

    if 'print_cov' in kwargs and kwargs['print_cov']:
        print('Last ',sliding_window,' samples: estimated covariance')
        print(est_sigma[-1])
    
    plt.show()

# Take the autocorrelation of a 1D array
def autocorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result[result.size/2:]


    