import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from train import *
from model_nn import Koopman
from data_loader import produce_dataset, get_dataloaders
from test import *
import numpy as np
import torch

if __name__ == '__main__':
    initial_out = sys.stdout
    # create dataset (optional)
    x_range = np.array([1.,1.,1.,0.5,0.5,1.5])
    u_range = np.array([0.5,0.5,0.5])
    SimLength=20
    Ntraj = 50000
    Ts=0.01
    path = produce_dataset(x_range,u_range,SimLength,Ntraj,Ts)
    sys.stdout = initial_out
    print(path)
    
    # get dataloaders
    #path = './dataset/x-[0.1 0.1 0.1 0.5 0.5 1.5]_u-[0.1 0.1 0.1]_0.1x20_Ts_0.01'
    train_loader, val_loader, test_loader = get_dataloaders(path)

    # trainers parameter setting
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # create trainer
    model = Koopman(n=3,K=10)
    hyper = [1.0,1.0,0.3,0.000000001,0.000000001,0.000000001,1]
    loss_function = Loss(hyper[0], hyper[1], hyper[2], hyper[3], hyper[4], hyper[5], hyper[6])

    # start training
    file_name = train_the_model(device, model, loss_function, train_loader, val_loader, hyper,epochs=100)
    plot_learning_curve(file_name)

    # get test result
    test_the_model(test_loader, model, loss_function, file_name)
    print(file_name)
    
    # see single trajectory(optional)
    result_sample(path,model,file_name,index=10)