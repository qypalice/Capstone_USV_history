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
x_min = np.array([-2,-2])
x_max = np.array([2,2])
u_min = np.array([-1.5,-0.5])
u_max = np.array([1.5,0.5])
du_min = np.array([-1.,-0.3])
du_max = np.array([1.,0.3])
#eps_min = np.array([-1.,-0.3])
#eps_max = np.array([1.,0.3])

def simulate_path(init_x,SimLength):
    # initialize
    X = np.zeros((2,SimLength+1))
    X[:,0] = init_x.squeeze()[:-1]
    # several step 
    s1 = SimLength/4
    s2 = SimLength/2
    s3 = 3*SimLength/4
    for i in range(SimLength-1):
        if i<s1:# go east
            X[:,i+1] = X[:,i]+0.8*np.array([1,0])
        elif i<s2:# go north
            X[:,i+1] = X[:,i]+0.8*np.array([0,1])
        elif i<s3:# go west
            X[:,i+1] = X[:,i]+0.6*np.array([-1,0])
        else:# go south
            X[:,i+1] = X[:,i]+0.6*np.array([0,-1])
        X[:,i+1] = np.maximum(X[:,i+1],x_min)
        X[:,i+1] = np.minimum(X[:,i+1],x_max)
    path = f'./dataset/MPC/SimLenth_{str(SimLength)}_Ts_{str(Ts)}'
    if not os.path.exists(path):
      os.makedirs(path)
    sim = {}
    sim['init state'] = init_x
    sim['path'] = X
    np.save(path,sim)
    print(path)

    return path

def MPC_solver(Q,R,rho,A,B,Yref,x,u,Nc):
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
        obj += cp.quad_form(Y[:,t+1]-Yref,Q) + cp.quad_form(U[:,t+1]-U[:,t],R)
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
    # plot the prediction
    pred = np.zeros((11,Nc))
    for i in range(Nc):
        pred[:,i] = Y[:,i+1].value
    MPC_lifted_plot(Yref,pred,Nc)
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
    for i in range(1,5):#path.shape[1]-Nc):
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