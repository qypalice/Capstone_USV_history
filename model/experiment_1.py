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
    batch_size = 1000
    train_loader, val_loader, test_loader = get_dataloaders(path,batch_size)

    epochs = 300
    model_num = 0
    # Varying steps
    for i in [5,10]:
        # get parameters
        model_num += 1
        K = 6
        arg = {
            'encoder':[3,32,64,K],
            'decoder':[K+3,128,64,32,3],
            'hyper':[1.0,1.0,0.3,0.000000001,0.000000001,0.000000001,i]
        }
        # start training
        print('Model '+str(model_num)+':')
        file_name = train_the_model(train_loader, val_loader, arg, batch_size, epochs)
        # get test result
        test_the_model(test_loader, file_name)
        print(file_name)
    