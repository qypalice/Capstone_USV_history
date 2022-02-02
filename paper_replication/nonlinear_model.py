from numpy import cos,sin
import numpy as np

#=========================================================================================
# discrete nonlinear model
#======================================================================================
def discrete_nonlinear(x,tau,Ts):
    """
    discrete_nonlinear(x,tau,Ts) returns the nonlinear continuous model
    x[k+1] = A*x[k] + B*tau of the state vector: 

    input vector    x   = [u v r]'
                    tau = [X, Y, N]' control force/moment
                    Ts    Sampling period
    output vector y = x

    u     = surge velocity                    (m/s)     
    v     = sway velocity                     (m/s)
    r     = yaw velocity                      (rad/s)

    matrix A = [-inv(M)*C_rb 0
                R            0]
           B = [inv(M) 0]
    """
    # Inputs
    state = x.squeeze()
    u = state[0]
    v = state[1]
    
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

    # Check of input and state dimensions
    if x.shape[0]  != 3:
        print('x-vector must have dimension 3 !')
    if tau.shape[0]  != 3:
        print('u-vector must have dimension 3 !')
        
    # get A,B
    A = np.zeros((3,3))
    A = -K*np.matmul(inv_M,C_rb)
    #print(A)
    #print(A)
    B = np.zeros((3,3))
    B = inv_M
    #print(B)
    #append(append(-K*np.linalg.inv(M).dot(C_rb),R,axis=))
    
    return Ts*(A.dot(x) + B.dot(tau))+x