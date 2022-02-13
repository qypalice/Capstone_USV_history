import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import Checkpoint
import sys


def test_the_model(test_loader, model, loss_function, file_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
    checkpoint = Checkpoint(saved_model_path)
    model=checkpoint.load_saved_model(model)
    model = model.to(device)
    model.eval()
    
    loss_avg = 0.
    progress_bar = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(progress_bar):
            X, U=data
            X = X.to(device)
            U = U.to(device)
            loss = loss_function(model,X,U)      
            loss_avg +=loss.item()
    loss_avg = loss_avg / (i + 1)
    print(f'\nLoss score: {str(loss_avg)}.')
    f = open('./results/{}.txt'.format(file_name), 'w')
    stdo = sys.stdout
    sys.stdout = f
    print(f'Loss score: {str(loss_avg)}.')
    f.close()
    sys.stdout = stdo

def get_prediction(X,U, model, file_name):
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
    checkpoint = Checkpoint(saved_model_path)
    model=checkpoint.load_saved_model(model)
    model = model.to(device)
    model.eval()
    # get prediction
    X = X.to(device)
    U = U.to(device)
    Y =  model(X,U)
    return Y

def position_plot(pred,truth,model_name):
    pred = pred.squeeze()
    truth = truth.squeeze()
    N = pred.shape[0]
    t = np.linspace(1,N,N)
    
    if model_name =='damping':
        plt.figure(figsize=(9,4.5))
        plt.subplot(121)
        plt.plot(t,truth[:,1],t,pred[:,1])
        plt.grid(True)
        plt.title('Position change')
        plt.xlabel('Time t')
        plt.ylabel('distance (m)')
        plt.legend(['nonlinear','linear'])

        plt.subplot(122)
        plt.plot(t,truth[:,0],t,pred[:,0])
        plt.grid(True)
        plt.title('Speed change')
        plt.xlabel('Time t')
        plt.ylabel('Speed (m/s)')
        plt.legend(['nonlinear','linear'])
        plt.show()

def result_sample(data_path,model_name, model,file_name,index=0):
    # get data
    xx = np.load(data_path+"/X_test.npy")
    uu = np.load(data_path+"/U_test.npy")
    #xx = np.load(data_path+"/X_train.npy")
    #uu = np.load(data_path+"/U_train.npy")
    xx = xx[index]
    uu = uu[index]
    #U = torch.tensor(uu).float()
    
    # get prediction
    yy = np.empty((xx.shape[0],2))
    yy[0] = xx[0]
    Y = torch.tensor(np.atleast_2d(xx[0])).float()
    for i in range(1,xx.shape[0]):
        U = torch.tensor(np.atleast_2d(uu[i-1])).float()
        Y = get_prediction(Y,U, model, file_name)
        yy[i] = Y.cpu().detach().numpy().squeeze()
    position_plot(yy,xx,model_name)
