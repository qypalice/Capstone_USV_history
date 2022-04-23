import numpy as np
from control import dlqr
from Koopman_numpy import Koopman_numpy
from MPC import MPC_process_plot

def dlqr_process(model_file,init,ref,Q,R,thre):
    #load model
    operater = Koopman_numpy(model_file)
    A,B = operater.linear_matrix()
    L = A.shape[0]

    # generate reference,initialization
    x = operater.encode(init)
    r = np.zeros(L)#operater.encode(ref)
    ref = operater.decode(r)
    path = np.zeros((3,0))
    ref_path = np.zeros((3,0))
    path = np.c_[path,init]
    ref_path = np.c_[ref_path,ref]
    lifted_path = np.zeros((L,0))
    lifted_ref_path = np.zeros((L,0))
    lifted_path = np.c_[lifted_path,x]
    lifted_ref_path = np.c_[lifted_ref_path,r]

    # get dlqr solution
    K, S, E = dlqr(A, B, Q, R)

    # start control process
    i = 0
    err = np.linalg.norm(x-r)
    print("Init state:"+str(init))
    print("Target state:"+str(ref))
    while err>thre and i<100000:
        u = -K@x
        x = A@x+B@u
        x = x.squeeze()
        path = np.c_[path,operater.decode(x)]
        ref_path = np.c_[ref_path,ref]
        lifted_path = np.c_[lifted_path,x]
        lifted_ref_path = np.c_[lifted_ref_path,r]
        err = np.linalg.norm(x-r)
        print("Step "+str(i+1)+": "+str(path[:,-1])+", "+str(u)+", "+str(err))
        i += 1
    MPC_process_plot(ref_path,path,path.shape[1],lifted=False)
    MPC_process_plot(lifted_ref_path,lifted_path,path.shape[1],lifted=True)
    



    