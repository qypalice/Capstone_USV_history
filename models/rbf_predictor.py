#====================================================================================
# linearization using rbf
#=================================================================================
from math import pi
from numpy.random import rand
from nonlinear_model import discrete_nonlinear
import matplotlib.pyplot as plt
import numpy as np
import os

def build_predictor(N,Ts,option):
    """
    This function returns the center of lift and parameters of linear matrices
    """
    vel = 0.1
    pos = 0.5
    angle = pi
    
    # get trajectories
    X_c,X,Y,U = get_trajectories_rbf(vel,pos,angle,N,Ts,option)


    # Build predictor
    # Regression for Matrix M = [A,B] (Eq. (22) in the paper)
    phi = np.r_[X,U]
    W = np.dot(phi,phi.T)
    V = np.dot(Y,phi.T)
    M = np.dot(V,np.linalg.pinv(W))
    
    # save the matrix
    filename = f"_order_{str(N)}_Ts_{str(Ts)}_option_{option}.npy"
    np.save("./rbf_model/M"+filename,M)
    np.save("./rbf_model/X_c"+filename,X_c)
    
def get_trajectories_rbf(vel,pos,angle,N,Ts,option):
    # Collect data
    SimLength = 10  # Number of steps per trajectory
    Ntraj = 10000
    
    # give the set of random input signals
    X_max = 0.1
    X_min = -X_max # define limit on forward force
    N_max = 0.1
    N_min = -N_max # define limit on rotation torque
    Ubig= rand(3,SimLength,Ntraj)
    Ubig[0,:,:] = Ubig[0,:,:]*(X_max-X_min)+X_min
    Ubig[1,:,:] = Ubig[1,:,:]*(X_max-X_min)+X_min
    Ubig[2,:,:] = Ubig[2,:,:]*(N_max-N_min)+N_min

    # run and collect data
    X = np.empty((6,0))
    Y = np.empty((6,0)) 
    U = np.empty((3,0))   # initialize
    plt.figure()
    for i in range(Ntraj):
        # Intial state is a random vector
        # The velocity is uniformly distributed in [-vel,vel]
        # psi is uniformly distributed in [-angle,angle]
        x = np.r_[rand(1,1)*2*vel-vel,
            rand(1,1)*2*vel-vel,
            rand(1,1)*2*vel-vel,
            rand(1,1)*2*pos-pos,
            rand(1,1)*2*pos-pos,
            rand(1,1)*2*angle-angle].squeeze()
        #x = np.array([0,0,0,0,0,0]).T
        xx = np.array(x)

        # Simulate one trajectory
        print('Trajectory {} out of {}'.format(i,Ntraj))
        for j in range(SimLength):
            x = discrete_nonlinear(x,Ubig[:,j,i],Ts)
            xx = np.c_[xx,x]
            U  = np.c_[U,Ubig[:,j,i]]
        plt.plot(xx[3,:],xx[4,:],zorder=1)
        # Store
        X = np.c_[X,xx[:,:-1]]
        Y = np.c_[Y,xx[:,1:]]
    print("Trajetories got, now lift the states...")

    
    # update X_c and get the trajectoryies again
    state_min = X.min(axis=1)
    state_range = X.max(axis=1)-state_min
    
    u = state_range[0]*rand(1,N)+state_min[0]
    v = state_range[1]*rand(1,N)+state_min[1]
    r = state_range[2]*rand(1,N)+state_min[2]
    x = state_range[3]*rand(1,N)+state_min[3]
    y = state_range[4]*rand(1,N)+state_min[4]
    psi = state_range[5]*rand(1,N)+state_min[5]
    X_c = np.r_[u,v,r,x,y,psi]
    plt.scatter(X_c[3,:],X_c[4,:],zorder=2)
    #eps = state_range.max(axis=0)/N
    eps = 1
    plt.show()
    
    # get lifted states
    X = np.r_[X,np.zeros((N,X.shape[1]))]
    Y = np.r_[Y,np.zeros((N,Y.shape[1]))]
    for i in range(X.shape[1]):
        X[:,i] = lift(N,X[0:6,i],X_c,eps,option).squeeze()
        Y[:,i] = lift(N,Y[0:6,i],X_c,eps,option).squeeze()     
    return X_c,X,Y,U
        
def lift(N,x,X_c,eps=1,option='gauss'):
    """
    x is initial state, which is a column vector (6*1)
    X_c is the centers in the condition box (6*N)
    X_lift is the lifted state (6+N)*1
    """
    y = np.array([x]).T
    for i in range(N):
        r_square = np.dot(x-X_c[:,i],x-X_c[:,i]).sum()
        if option == 'gauss':
            temp = np.exp(-eps**2*r_square)
        elif option == 'NN':
            temp = np.exp(-eps**2*np.dot(x-X_c[:,i],x-X_c[:,i]).sum())
        else: # use thinplate
            temp = np.dot(r_square,np.log(np.sqrt(r_square)))
        y = np.r_[y,np.array([[temp]])]
    return y

def load_linear_model(N,Ts,option):
    filename = f"_order_{str(N)}_Ts_{str(Ts)}_option_{option}.npy"
    if not os.path.isfile("./M"+filename):
        build_predictor(N,Ts,option)
    M = np.load("./rbf_model/M"+filename)
    X_c = np.load("./rbf_model/X_c"+filename)
    A = M[:,0:N+6]
    B = M[:,N+6:]
    C = np.c_[np.ones((1,6)),np.zeros((1,N))]
    return A,B,C,X_c
