from numpy import cos,sin
import numpy as np

#=========================================================================================
# discrete nonlinear model
#======================================================================================
def discrete_nonlinear(x,tau,Ts,model_name='damping'):
    """
    discrete_nonlinear(x,tau,Ts) returns the nonlinear continuous model
    x[k+1] = A*x[k] + B*tau of the state vector: 

    'damping' system:
    F-kx-qv^2 = m dv
    input vector    x   = [v x]' velocity (m/s) and distance (m)
                    tau = [F]' control force (N)
                    Ts    Sampling period
    output vector   y = x

    matrix A = [qv/m    -k/m
                1          0]
           B = [1/m 0]'
    """
    # Inputs
    state = x.squeeze()
    if model_name == 'damping':
        v = state[0]
        # parameters
        m = 0.8
        k = 300
        q = 10
        # Model matricses
        A = np.array([[-q*v/m,-k/m],[1,0]])
        B = np.array([[1/m],[0]])
    
    return Ts*(A.dot(x) + B.dot(tau))+x