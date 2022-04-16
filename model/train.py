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
        self.best = None

    def early_stopping(self, loss, model):

        if self.best is None:
            self.best = loss
        elif self.best > loss:
            self.best = loss
            self.num_bad_epochs = 0
            self.save(model)
        else:
            self.num_bad_epochs += 1

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
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adadelta(self.model.parameters(),lr=0.1, rho=0.8)

    def train(self,  epochs,  checkpoint,  csv_logger):
        # initialize parameters
        self._weight_init(self.model)

        # start training
        patience = min(int(epochs*0.5),20)
        train_loss = []
        val_loss  = []    

        for epoch in range(epochs):
            loss_tra= self.train_one_epoch(epoch)
            loss_tra= self.validate(self.train_loader)
            train_loss.append(loss_tra)
            
            loss_val= self.validate(self.val_loader)
            val_loss.append(loss_val)
            
            tqdm.write('val_loss: %.3f' % (loss_val))
            row = {'epoch': str(epoch+1), 'train_loss': str(loss_tra), 'val_loss': str(loss_val)}
            
            csv_logger.writerow(row)
            checkpoint.early_stopping(loss_val, self.model)
            if checkpoint.num_bad_epochs>=patience:
                tqdm.write("Early stopping with {:.3f} best score, the model did not improve after {} iterations".format(
                        checkpoint.best, checkpoint.num_bad_epochs))
                break
        csv_logger.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_avg = 0.
        progress_bar = tqdm(self.train_loader)
        
        # show progress
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch+1))
            
            X, U=data
            X = X.to(self.device)
            U = U.to(self.device)
            
            self.model.zero_grad()
            loss = self.loss_function(self.model,X,U)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
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
            init.xavier_uniform_(m.weight.data,gain=nn.init.calculate_gain('relu'))
            init.normal_(m.bias.data)


def start_logging(filename):
    f = open('./logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()

def train_the_model(train_loader, val_loader, arg, batch_size=10,epochs=200):
    # create trainer
    hyper = arg['hyper']
    en = arg['encoder']
    de = arg['decoder']

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Koopman(en,de)
    loss_function = Loss(hyper[0], hyper[1], hyper[2], hyper[3], hyper[4], hyper[5], hyper[6])

    trainer = Trainer(device, model, loss_function, train_loader, val_loader, batch_size)
    print(f'Trainer created.', end = "\n")

    # define file names
    file_name=f"encoder_{str(en)}_decoder_{str(de)}_hyper_{str(hyper)}_batch_{str(batch_size)}"

    csv_logger = CSVLogger(filename=f'./logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'val_loss'])
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)

    checkpoint = Checkpoint(saved_model_path)

    # initialize recording
    experiment_name = file_name+'_{}'.format(datetime.utcnow().strftime('%m-%d-%H-%M'))
    stdo = sys.stdout
    f = start_logging(experiment_name)
    print(f'Starting {experiment_name} experiment')

    # start training
    trainer.train(epochs, checkpoint, csv_logger)
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

def save_model_as_numpy(file_name):
    # get parameters
    arguments = file_name.split('_')
    en = list(map(int,arguments[1][1:-1].split(', ')))
    de = list(map(int,arguments[3][1:-1].split(', ')))
    hyper = list(map(float,arguments[5][1:-1].split(', ')))

    # set model
    model = Koopman(en,de)
    saved_model_path = './weight/{}_checkpoint.pt'.format(file_name)
    checkpoint = Checkpoint(saved_model_path)
    model=checkpoint.load_saved_model(model)
    
    # create dictionary
    param = {}
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        param[name]=parameters.detach().numpy()
    
    # save as numpy file\
    np.save('./numpy_weight/'+file_name,param)