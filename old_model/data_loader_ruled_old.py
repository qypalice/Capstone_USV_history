import torch
import numpy as np
from tqdm import tqdm
from numpy.random import rand,randint
from nonlinear_model import discrete_nonlinear
import os

def simulation(noise_range,dest_range,K,SimLength=10,Ntraj=1000,Ts=0.01):
    '''
    This function is to get simulated trajectories of the USV, and save 
    the trajectories and relevant input in numpy file. 

    Input Parameters:
    noise_range     a small float to show the noise range of the initial state
    dest_range      a large 3*1 numpy array, to show the range of the destination
    K               The proportional feedback controller parameter
    SimLength       the length of the trajectory, unit is the sampling period
    Ntraj           the number of trajectories, i.e. size of the dataset
    Ts              sampling period

    Return:
    filename is to provide the parameter information of the dataset

    Simulation rules:
    (i)  For input, assume error is "e", input is K*e
    (ii) For initial state, it is fixed with noise added.(Vary if needed, TBA)
    '''
    # get the range for input
    #Ubig= rand(3,SimLength,Ntraj)
    #Ubig[0,:,:] = Ubig[0,:,:]*2*u_range[0]-u_range[0]
    #Ubig[1,:,:] = Ubig[1,:,:]*2*u_range[1]-u_range[1]
    #Ubig[2,:,:] = Ubig[2,:,:]*2*u_range[2]-u_range[2]
    #Ubig = randint(4, size=(3,SimLength,Ntraj))+ rand(3,SimLength,Ntraj)*u_var-1
    
    # produce a series of destination
    dest = randint(2, size=(Ntraj,3))*2-1
    dest[:,0] = dest[:,0]*dest_range[0]
    dest[:,1] = dest[:,1]*dest_range[1]
    dest[:,2] = dest[:,2]*dest_range[2]

    # run and collect data
    X = np.empty((Ntraj,SimLength+1,6))
    U = np.empty((Ntraj,SimLength,3))   # initialize
    pbar = tqdm(total=Ntraj)
    for i in range(Ntraj):
        xx = np.empty((SimLength+1,6))
        # Intial state is a random vector within given range
        x = np.zeros((1,6))+rand(1,6)*noise_range
        x = x.squeeze()
            #np.r_[rand(1,1)*2*x_range[0]-x_range[0],
                #rand(1,1)*2*x_range[1]-x_range[1],
                #rand(1,1)*2*x_range[2]-x_range[2],
                #rand(1,1)*2*x_range[3]-x_range[3],
                #rand(1,1)*2*x_range[4]-x_range[4],
                #rand(1,1)*2*x_range[5]-x_range[5]].squeeze()
        xx[0,:] = x

        # Simulate one trajectory
        for j in range(SimLength):
            u = K*(dest[i,:].squeeze()-x[3:6])+noise_range*rand(1,3).squeeze()
            x = discrete_nonlinear(x,u,Ts)
            xx[j+1,:] = x
            U[i,j,:] = u
        # Store
        X[i,:,:] = xx
        pbar.update(1)
    pbar.close()
    return X,U

def produce_dataset(noise_range,dest_range,K,SimLength=10,Ntraj=1000,Ts=0.01):
    print("Start simulating...")
    X_train,U_train = simulation(noise_range,dest_range,K,SimLength,int(0.6*Ntraj),Ts=0.01)
    X_val,U_val = simulation(noise_range,dest_range,K,SimLength,int(0.2*Ntraj),Ts=0.01)
    X_test,U_test = simulation(noise_range,dest_range*2,K,SimLength,int(0.2*Ntraj),Ts=0.01)

    # save the matrix
    print("\nDataset produced.")
    path = f'./dataset/noise-{str(noise_range)}_dest-{str(dest_range)}_K-{str(K)}_{str(SimLength*Ts)}x{str(Ntraj)}_Ts_{str(Ts)}'
    if not os.path.exists(path):
      os.makedirs(path)
    np.save(path+"/X_train",X_train)
    np.save(path+"/U_train",U_train)
    np.save(path+"/X_val",X_val)
    np.save(path+"/U_val",U_val)
    np.save(path+"/X_test",X_test)
    np.save(path+"/U_test",U_test)
    print("Dataset saved.")
    return path

def get_data(path):
    xx_train = np.load(path+"/X_train.npy")
    xx_val = np.load(path+"/X_val.npy")
    xx_test = np.load(path+"/X_test.npy")
    uu_train = np.load(path+"/U_train.npy")
    uu_val = np.load(path+"/U_val.npy")
    uu_test = np.load(path+"/U_test.npy")
    X_train = torch.tensor(xx_train).float()
    U_train = torch.tensor(uu_train).float()
    X_val = torch.tensor(xx_val).float()
    U_val = torch.tensor(uu_val).float()
    X_test = torch.tensor(xx_test).float()
    U_test = torch.tensor(uu_test).float()
    return  X_train,U_train,X_val,U_val,X_test,U_test      

#==========================================================================
# Dataset loader
#==========================================================================
class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,U):
        self.X  = X
        self.U  = U
        
    def __len__(self):
        return self.U.shape[0]

    def __getitem__(self, index):
        xx = self.X[index]
        uu = self.U[index]
        return xx,uu
        
        
def get_loaders(X,U,batch_size=1):
    data = Dataset(X,U)
    loader=torch.utils.data.DataLoader(data, batch_size, shuffle=True, num_workers=2,drop_last=False)
    return loader

def get_dataloaders(path):
    X_train,U_train,X_val,U_val,X_test,U_test = get_data(path)

    # get dataloaders
    tra_loader = get_loaders(X_train,U_train)
    val_loader = get_loaders(X_val,U_val)
    test_loader = get_loaders(X_test,U_test)
    return tra_loader, val_loader, test_loader