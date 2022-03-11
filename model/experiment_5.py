import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from train import *
from data_loader import produce_dataset, get_dataloaders
from test_function import *
import numpy as np

if __name__ == '__main__':
    initial_out = sys.stdout
    # get dataloaders
    path = './dataset/state-[2. 2. 3.]_input-[1.5 0.5]_2.0x100000_Ts_0.1'
    batch_size = 1
    train_loader, val_loader, test_loader = get_dataloaders(path,batch_size)

    epochs = 300
    # get parameters
    K = 8
    arg = {
        'encoder':[3,32,64,K],
        'decoder':[K+3,128,64,32,3],
        'hyper':[1.0,3.0,0.3,1e-7,1e-7,1e-7,10]
    }
    # start training
    file_name = train_the_model(train_loader, val_loader, arg, batch_size, epochs)
    # get test result
    test_the_model(test_loader, file_name)
    print(file_name)
    