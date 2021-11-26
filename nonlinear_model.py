from numpy import cos,sin,pi
import numpy as np

#=========================================================================================
# discrete nonlinear model
#======================================================================================
def discrete_nonlinear(x,tau,Ts):
    """
    discrete_nonlinear(x,tau,Ts) returns the nonlinear continuous model
    x[k+1] = A*x[k] + B*tau of the state vector: 

    input vector    x   = [u v r N E psi]'
                    tau = [X, Y, N]' control force/moment
                    Ts    Sampling period
    output vector y = x

    u     = surge velocity                    (m/s)     
    v     = sway velocity                     (m/s)
    r     = yaw velocity                      (rad/s)
    N     = position in x-direction           (m)
    E     = position in y-direction           (m)
    psi   = yaw angle                         (rad)

    matrix A = [-inv(M)*C_rb 0
                R            0]
           B = [inv(M) 0]
    """
    # Inputs
    state = x.squeeze()
    u = state[0]
    v = state[1]
    psi = state[5]
    
    # Normalization variables
    m = 6  # mass (kg)
    Iz = 2 # inertial
    K = 1  # correlation factor (in theory 1)

    # Model matricses
    inv_M = np.array([[1/m,0,0],[0,1/m,0],[0,0,1/Iz]])
    #print(inv_M)
    C_rb = np.array([[0,0,-m*v],[0,0,m*u],[m*v,-m*u,0]])
    #print(C_rb)
    #print(np.matmul(inv_M,C_rb))
    R = np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]])
    
    # Check of input and state dimensions
    if x.shape[0]  != 6:
        print('x-vector must have dimension 6 !')
    if tau.shape[0]  != 3:
        print('u-vector must have dimension 3 !')
        
    # get A,B
    A = np.zeros((6,6))
    A[0:3,0:3] = -K*np.matmul(inv_M,C_rb)
    #print(A)
    A[3:6,0:3] = R
    #print(A)
    B = np.zeros((6,3))
    B[0:3,:] = inv_M
    #print(B)
    #append(append(-K*np.linalg.inv(M).dot(C_rb),R,axis=))
    
    return Ts*(A.dot(x) + B.dot(tau))+x