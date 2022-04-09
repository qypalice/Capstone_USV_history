#from casadi import *
import os
import sys
from tabnanny import verbose
import cvxpy as cp
from inspect import GEN_RUNNING
import numpy as np
from numpy.random import rand
from numpy.linalg import matrix_power
from numpy import pi
import matplotlib.pyplot as plt
from nonlinear_model import discrete_nonlinear

# define global parameters
Ts = 0.1
u_min = 10*np.array([-1.5,-0.5])
u_max = 10*np.array([1.5,0.5])
du_min = 10*np.array([-1.,-0.3])
du_max = 10*np.array([1.,0.3])
#eps_min = np.array([-1.,-0.3])
#eps_max = np.array([1.,0.3])

def simulate_path(SimLength):
    # initialize
    X = np.zeros((3,SimLength+1))
    U = np.zeros((2,SimLength))

    '''# define u
    s1 = SimLength/3
    s2 = SimLength/3
    s3 = SimLength/3
    for i in range(SimLength-1):
        if i<s1:
            U[:,i+1] = U[:,i]+0.05*np.multiply(du_max,rand(2))
        elif i<s2:
            U[:,i+1] = U[:,i]+0.05*np.multiply(du_min,rand(2))
        elif i<s3:
            U[:,i+1] = U[:,i]+0.05*np.array([du_min[0]*rand(),du_max[1]*rand()])
        else:
            U[:,i+1] = U[:,i]+0.05*np.array([du_max[0]*rand(),du_min[1]*rand()])
        U[:,i+1] = np.maximum(U[:,i+1],u_min)
        U[:,i+1] = np.minimum(U[:,i+1],u_max)

    # start simulation as save state
    for i in range(SimLength):
        X[:,i+1] = discrete_nonlinear(X[:,i],U[:,i],Ts)'''
    X = 0.3*np.array([np.cos(np.linspace(1*pi, 0, num=SimLength)),np.sin(np.linspace(1*pi, 0, num=SimLength))-1])
    path = f'./dataset/MPC/SimLenth_{str(SimLength)}_Ts_{str(Ts)}'
    if not os.path.exists(path):
      os.makedirs(path)
    np.save(path+"/X",X[:2,:])
    print(path)

    return path

def get_Augmented_Matrix(A,B,Q,R,rho,Np,Nc):
    """
    input:
    A,B     matrices of linear system
    Q,R     penalty matrices for MPC
    rho     v
    """
    '''determine matrices when input is delta u instead of u
    A_bar = [A      B
            O(m*L)  I(m*m)]
    B_bar = [B
            I(m*m)]
    C = [I(L*L) O(L*m)]
    state = [lifted state
            input u]
    '''
    L = A.shape[0]
    m = B.shape[1]
    A_bar = np.zeros((m+L,m+L))
    A_bar[0:L,0:L] = A
    A_bar[0:L,L:] = B
    A_bar[L:,L:] = np.eye(m)
    B_bar = np.r_[B,np.eye(m)]
    C = np.zeros((L,m+L))
    #C[:2,:2] = np.eye(2)
    C[:L,:L] = np.eye(L)

    # get more straight forward matrices for MPC (more steps)
    Gamma = C @ A_bar
    Qbig = np.zeros((L*Np,L*Np))
    Qbig[0:L,0:L] = Q
    for i in range(1,Np):
        Gamma = np.r_[Gamma,C @ matrix_power(A_bar,i+1)]
        Qbig[i*L:(i+1)*L,i*L:(i+1)*L] = Q

    Theta_r = C @ B_bar
    for i in range(1,Np-Nc+1):
        Theta_r = np.r_[Theta_r,(C @ matrix_power(A_bar,i)) @ B_bar]
    Theta = np.r_[np.zeros((L*(Nc-1),m)),Theta_r]
    Rbig = np.zeros((m*Nc,m*Nc))
    Rbig[0:m,0:m] = R
    for i in range(1,Nc):
        Rbig[i*m:(i+1)*m,i*m:(i+1)*m] = R
        Theta_r = np.r_[Theta_r,(C @ matrix_power(A_bar,Np-Nc+i)) @ B_bar]
        Theta = np.c_[np.r_[np.zeros((L*(Nc-1-i),m)),Theta_r],Theta]
    
    # calculate penalty matrix
    rho = rho*np.eye(m*(Np-Nc))
    H = np.r_[np.c_[Theta.T @ Qbig @ Theta+Rbig,np.zeros((m*Nc,m*(Np-Nc)))],
            np.c_[np.zeros((m*(Np-Nc),m*Nc)),rho]]
    return Gamma,Theta,Qbig,H

def MPC_solver(Q,R,rho,A,B,Yref,x,u,Nc):
    m = 2
    # define variables -- the combination of dU and slack variable eps
    #group = SX.sym('dU',Np*2)
    U = cp.Variable((2,Nc+1))
    Y = cp.Variable((11,Nc+1))
    eps = cp.Variable(2)

    # define object function
    obj = rho * cp.sum_squares(eps)

    # define constraints
    cons = [eps>=0,Y[:,0]==x,U[:,0]==u]
    for t in range(Nc):
        obj += cp.quad_form(Y[:,t+1]-Yref[:,t],Q) + cp.quad_form(U[:,t+1]-U[:,t],R)
        cons += [Y[:,t+1] == A@Y[:,t] + B@U[:,t+1],
                u_min<=U[:,t+1], u_max>=U[:,t+1],
                du_min-eps<=U[:,t+1]-U[:,t], du_max+eps>=U[:,t+1]-U[:,t]]
    # define solver
    prob = cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-6,verbose=False)
    #print(prob.status)
    print(eps.value)
    u = U[:,1].value
    y = Y[:,1].value
    return u,y


from Koopman_numpy import Koopman_numpy
import time
def MPC_control_process(model_file,path_ref,init_input,Q,R,rho,Nc): #temp
    #load model
    operater = Koopman_numpy(model_file)
    A,B = operater.linear_matrix()
    L = A.shape[0]

    # initialization
    file_name = f'Q-{str(np.diag(Q))}_R-{str(np.diag(R))}_rho-{str(rho)}_Nc-{str(Nc)}'
    diff = path_ref[:,1:]-path_ref[:,:-1]
    angle = np.arctan2(diff[1,:],diff[0,:])
    path_ref = np.r_[path_ref[:,:-1],np.array([angle])]
    path = path_ref.copy()

    lifted_ref = np.zeros((L,path_ref.shape[1]))
    for i in range(path_ref.shape[1]):
        lifted_ref[:,i] = operater.encode(path_ref[:,i])
    u = init_input
    y = lifted_ref[:,0]
    lifted_path = lifted_ref.copy()
    t_avg = 0
    # start contorl simulation
    for i in range(1,path.shape[1]-Nc):
        print('Step '+str(i)+' - MSE error in lifted space,state x, input u:')
        T1 = time.perf_counter()
        u,y = MPC_solver(Q,R,rho,A,B,lifted_ref[:,i:i+Nc],y,u,Nc)
        T2 = time.perf_counter()
        t_avg += T2-T1
        #path[:,i] = discrete_nonlinear(path[:,i-1],u,Ts)
        #lifted_x = operater.linear(operater.encode(path[:,i-1]),u)
        lifted_path[:,i] = y
        path[:,i] = operater.decode(y)
        if path[2,i] > pi:
            path[2,i] -= 2*pi
        elif path[2,i] < -pi:
            path[2,i] += 2*pi
        
        print(np.square(y-lifted_ref[:,i+1]).mean(),path[:,i],u)
    MPC_lifted_plot(lifted_ref,lifted_path,path.shape[1]-Nc)
    np.save('./results/MPC/{}'.format(file_name),path)

    # plot
    t = np.linspace(1,path.shape[1]-Nc,path.shape[1]-Nc)
    legend_list = ['truth','control']
    plt.figure(figsize=(5,5))
    plt.plot(t,path_ref[2,:-Nc],'o-')
    plt.plot(t,path[2,:-Nc])
    plt.grid(True)
    plt.xlabel('Time t')
    plt.ylabel('Theta')
    plt.title('Angle change')
    plt.legend(legend_list)
    plt.show()
    # see the time consumption
    t_avg /= path.shape[1]-Nc-1
    t_avg *= 1000
    print("Average time needed per step is "+str(t_avg)+" ms.")
    # see the control result
    err = np.linalg.norm(path-path_ref)**2 / (path.shape[1]-Nc-1)
    print("MSE loss: "+str(err))
    print('Controled path file: '+file_name)
    stdo = sys.stdout
    f = open('./results/MPC/{}.txt'.format(file_name), 'w')
    sys.stdout = f
    print(f'\nMSE loss: {str(err)}.')
    print("Average time needed per step is "+str(t_avg)+" ms.")
    f.close()
    sys.stdout = stdo
    return file_name

def MPC_result_plot(ref_file,control_files,Np):
    preds = []
    for control_file in control_files:
        pred = np.load('./results/MPC/{}.npy'.format(control_file)).T
        pred = pred[:-Np,:]
        preds.append(pred)
    truth = np.load('./dataset/MPC/{}/X.npy'.format(ref_file)).T
    truth = truth[:-Np-1,:]

    N = truth.shape[0]
    t = np.linspace(1,N,N)
    legend_list = ['truth']
    for i in range(len(preds)):
        legend_list.append('contorl '+str(i+1))

    # plot
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.plot(t,truth[:,0],'o-')
    for i in range(len(preds)):
        plt.plot(t,preds[i][:,0])
    plt.grid(True)
    plt.xlabel('Time t')
    plt.ylabel('x direction')
    plt.title('X position change')
    plt.legend(legend_list)

    plt.subplot(132)
    plt.plot(t,truth[:,1],'o-')
    for i in range(len(preds)):
        plt.plot(t,preds[i][:,1])
    plt.grid(True)
    plt.grid(True)
    plt.title('Y position change')
    plt.xlabel('Time t')
    plt.ylabel('y direction')
    plt.legend(legend_list)

    plt.subplot(133)
    plt.plot(truth[:,0],truth[:,1],'o-')
    for i in range(len(preds)):
        plt.plot(preds[i][:,0],preds[i][:,1])
    plt.grid(True)
    plt.grid(True)
    plt.title('position change')
    plt.xlabel('x direction')
    plt.ylabel('y direction')
    plt.legend(legend_list)

    plt.show()

def MPC_lifted_plot(ref,control,N):
    t = np.linspace(1,N,N)
    legend_list = ['ref','control']

    # plot
    plt.figure(figsize=(16,12))
    for i in range(11):
        plt.subplot(3,4,i+1)
        plt.plot(t,ref[i,:N],'o-')
        plt.plot(t,control[i,:N])
        plt.grid(True)
        plt.xlabel('Time t')
        plt.legend(legend_list)
    plt.show()