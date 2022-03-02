import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import Checkpoint
from model_nn import Koopman,Loss
import sys


def test_the_model(test_loader, file_name):
    # get parameters
    arguments = file_name.split('_')
    en = list(map(int,arguments[1][1:-1].split(', ')))
    de = list(map(int,arguments[3][1:-1].split(', ')))

    # set model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Koopman(en,de)
    loss_function = torch.nn.MSELoss()
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
    checkpoint = Checkpoint(saved_model_path)
    model=checkpoint.load_saved_model(model)
    model = model.to(device)
    model.eval()
    
    #start calculation
    loss_avg = 0.
    progress_bar = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(progress_bar):
            X, U=data
            X = X.to(device)
            U = U.to(device)
            #loss = loss_function(model,X,U)
            Y = get_prediction(X,U,model)
            loss = loss_function(X,Y)
            loss_avg +=loss.item()
    loss_avg = loss_avg / (i + 1)

    #print result in kernal and txt file
    stdo = sys.stdout
    print(f'\nLoss score: {str(loss_avg)}.')
    f = open('./results/{}.txt'.format(file_name), 'w')
    sys.stdout = f
    print(f'Loss score: {str(loss_avg)}.')
    f.close()
    sys.stdout = stdo

def get_prediction(X,U, model):
    # get encoder,linear system, decoder
    submodules = []
    for idx, m in enumerate(model.children()):
        submodules.append(m)
    en = submodules[0]
    de = submodules[1]
    K = submodules[2]

    Y = X.clone()
    K_i_en_x = en(X[:,0,:])
    for i in range(1,X.shape[1]):
        K_i_en_x = K(K_i_en_x,U[:,i-1,:].clone())
        Y[:,i,:] = de(K_i_en_x)
    return Y

def position_plot(preds,truth):
    # get data and set parameters
    preds = preds
    truth = truth
    num_model = preds.shape[0]
    N = truth.shape[0]
    t = np.linspace(1,N,N)
    legend_list = ['non_linear']
    for i in range(num_model):
        legend_list.append('model '+str(i+1))

    # plot
    plt.figure(figsize=(9,4.5))
    plt.subplot(121)
    plt.plot(truth[:,0],truth[:,1])
    for i in range(num_model):
        plt.plot(preds[i,:,0],preds[i,:,1],'o-')
    plt.grid(True)
    plt.xlabel('y direction')
    plt.ylabel('x direction')
    plt.title('location change')
    plt.legend(legend_list)

    plt.subplot(122)
    plt.plot(t,truth[:,2])
    for i in range(num_model):
        plt.plot(t,preds[i,:,2],'o-')
    plt.grid(True)
    plt.grid(True)
    plt.title('Angle change')
    plt.xlabel('Time t')
    plt.ylabel('$\psi$ (rad)')
    plt.legend(legend_list)
    plt.show()

def result_sample(data_path,file_names,index=0):
    # get data
    xx = np.load(data_path+"/X_test.npy")
    uu = np.load(data_path+"/U_test.npy")
    #xx = np.load(data_path+"/X_train.npy")
    #uu = np.load(data_path+"/U_train.npy")
    xx = xx[index]
    uu = uu[index]
    yy = np.empty((len(file_names),xx.shape[0],xx.shape[1]))
    #uu = np.array([[0.5,0.5,0.5]])
    #U = torch.tensor(uu).float() 
    X = torch.tensor(np.atleast_2d(xx)).float().unsqueeze(0)#xx[0])).float()
    U = torch.tensor(np.atleast_2d(uu)).float().unsqueeze(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    U = U.to(device)
    
    i = 0
    for file_name in file_names:
        # load model
        arguments = file_name.split('_')
        en = list(map(int,arguments[1][1:-1].split(', ')))
        de = list(map(int,arguments[3][1:-1].split(', ')))
        model = Koopman(en,de)
        saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
        checkpoint = Checkpoint(saved_model_path)
        model=checkpoint.load_saved_model(model)
        model = model.to(device)
        model.eval()
        
        # get prediction
        Y = get_prediction(X,U, model)
        yy[i,:,:] = Y.cpu().detach().numpy().squeeze()
        i += 1
    position_plot(yy,xx)
