import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from data_loader import *
from model_nn import *
import csv
import os
import sys
from datetime import datetime
from abc import *
from tqdm import tqdm
import matplotlib.pyplot as plt

class Checkpoint(object):
    
    def __init__(self,filename: str=None):

        self.saved_model_path = filename
        self.num_bad_epochs = 0
        self.is_better = None
        self.best = None

    def save(self, model):
        """
        Save best models
        arg:
           state: model states
           is_best: boolen flag to indicate whether the model is the best model or not
           saved_model_path: path to save the best model.
        """
        print("model saved.")
        torch.save(model.state_dict(), self.saved_model_path)

        #torch.save(state, self.saved_model_path)

    def load_saved_model(self, model):
        saved_model_path = self.saved_model_path

        if os.path.isfile(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path))
        else:
            print("=> no checkpoint found at '{}'".format(saved_model_path))
            
        return model

class CSVLogger():
    def __init__(self, filename, fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        """
        writer = csv.writer(self.csv_file)
        
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])
        """

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()



class Trainer(metaclass=ABCMeta):
    def __init__(self, device, model, loss_function,
                train_loader, val_loader, batch_size=16):    
        # import param
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self,  epochs,  checkpoint,  csv_logger):
        # initialize parameters
        self._weight_init(self.model)

        # start training
        train_loss = []
        val_loss  = []    

        for epoch in range(epochs):
            loss_tra= self.train_one_epoch(epoch)
            loss_tra= self.validate(self.train_loader)
            train_loss.append(loss_tra)
            
            loss_val= self.validate(self.val_loader)
            val_loss.append(loss_val)
            
            tqdm.write('val_loss: %.3f' % (loss_val))
            row = {'epoch': str(epoch), 'train_loss': str(loss_tra), 'val_loss': str(loss_val)}
            
            csv_logger.writerow(row)
            
        csv_logger.close()
        checkpoint.save(self.model)

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_avg = 0.
        progress_bar = tqdm(self.train_loader)
        
        # show progress
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            
            X, U=data
            X = X.to(self.device)
            U = U.to(self.device)
            
            self.model.zero_grad()
            loss = self.loss_function(self.model,X,U)
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.optimizer.step()
            
            loss_avg +=loss.item()

            progress_bar.set_postfix(loss='%.3f' % (loss_avg / (i + 1)))
        return loss_avg / (i + 1)

    def validate(self, loader):
        self.model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    
        loss_avg = 0.
        
        with torch.no_grad():

            for i, data in enumerate(loader):
                
                X, U=data
                X = X.to(self.device)
                U = U.to(self.device)
                
                loss = self.loss_function(self.model,X,U)      

                loss_avg +=loss.item()
        self.model.train()
        return loss_avg / (i + 1)

    def _weight_init(self, m):    
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)

def get_dataloaders(path):
    X_train,U_train,X_val,U_val,X_test,U_test = get_data(path)

    # get dataloaders
    tra_loader = get_loaders(X_train,U_train)
    val_loader = get_loaders(X_val,U_val)
    test_loader = get_loaders(X_test,U_test)
    return tra_loader, val_loader, test_loader

def start_logging(filename):
    f = open('./logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()

def train_the_model(device, model, loss_function, train_loader, val_loader, hyper, hidden_layer = 2, epochs=200):
    # define parameters
    file_name=f"hyper_{str(hyper)}_hidden_layer_{str(hidden_layer)}"

    csv_logger = CSVLogger(filename=f'./logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'val_loss'])
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
    checkpoint = Checkpoint(saved_model_path)
    trainer = Trainer(device, model, loss_function, train_loader, val_loader, batch_size = 1)
    print(f'Trainer created.', end = "\n")
    
    # initialize recording
    experiment_name = file_name+'_{}'.format(datetime.utcnow().strftime('%m-%d-%H-%M'))
    stdo = sys.stdout
    f = start_logging(experiment_name)
    print(f'Starting {experiment_name} experiment')

    # start training
    trainer.train(epochs,  checkpoint, csv_logger)
    stop_logging(f)
    sys.stdout = stdo

    return file_name

def plot_learning_curve(file_name):
    # read data
    filename=f'./logs/{file_name}.csv'
    Data = np.loadtxt(open(filename),delimiter=",",skiprows=1)
    epoch = Data[:,0]
    train_loss = Data[:,1]
    val_loss = Data[:,2]
    
    # plot data
    labels = ['train_loss', 'val_loss']
    plt.figure()
    plt.plot(epoch, train_loss, color='r')
    plt.plot(epoch, val_loss, color='k')
    plt.legend(labels=labels)
    plt.show() 

def test_the_model(test_loader, model, loss_function, file_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
    checkpoint = Checkpoint(saved_model_path)
    model=checkpoint.load_saved_model(model)
    model.eval()
    
    loss_avg = 0.
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, U=data
            X = X.to(device)
            U = U.to(device)
            loss = loss_function(model,X,U)      
            loss_avg +=loss.item()
    loss_avg = loss_avg / (i + 1)

    print(f'Loss score: {str(loss_avg)}.')
    f = open('./results/{}.txt'.format(file_name), 'w')
    stdo = sys.stdout
    sys.stdout = f
    print(f'Loss score: {str(loss_avg)}.')
    f.close()
    sys.stdout = stdo