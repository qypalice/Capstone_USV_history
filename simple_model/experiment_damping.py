import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from train_new import *
from model_nn import Koopman,Loss
from data_loader import produce_dataset, get_dataloaders
from test_function import *
import numpy as np
import torch

if __name__ == '__main__':
    initial_out = sys.stdout
    # create dataset (optional)
    model_name = 'damping'
    '''SimLength=10
    Ntraj = 100000
    Ts=0.01
    path = produce_dataset(model_name,SimLength,Ntraj,Ts)
    sys.stdout = initial_out
    print(path)'''
    path = './dataset/damping_0.1x100000_Ts_0.01'
    
    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(path)

    # trainers parameter setting
    n = 2
    K = 14
    struct_encoder = [n,32,64,K,K+n]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # create trainer
    model = Koopman(n=2,K=7,u=1)
    hyper = [1.0,1.0,0.3,0.000000001,0.000000001,0.000000001,1]
    loss_function = torch.nn.MSELoss(reduction='sum')

    # start training
    file_name = train_the_model(device, model_name, model, loss_function, train_loader, val_loader, hyper,epochs=10, batch_size=10,P=hyper[6])
    plot_learning_curve(file_name)

    # get test result
    test_the_model(test_loader, model, loss_function, file_name)
    print(file_name)
    
    # see single trajectory(optional)
    result_sample(path,model_name,model,file_name,index=2)