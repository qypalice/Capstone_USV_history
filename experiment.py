from trainer import *
from model_nn import Koopman

# create dataset (optional)
x_range = np.array([0.1,0.1,0.1,0.5,0.5,1.5])
u_range = np.array([0.1,0.1,0.1])
SimLength=100
Ntraj = 200
Ts=0.01
path = produce_dataset(x_range,u_range,SimLength,Ntraj,Ts)

# get dataloaders
#path = './dataset/x-[0.1 0.1 0.1 0.5 0.5 1.5]_u-[0.1 0.1 0.1]_0.1x20_Ts_0.01'
train_loader, val_loader, test_loader = get_dataloaders(path)

# create trainers
device = "cuda:0" if torch.cuda.is_available() else "cpu"
hyper = [1.0,1.0,0.3,0.000000001,0.0000000001,0.000000001,20]
loss_function = Loss(hyper[0], hyper[1], hyper[2], hyper[3], hyper[4], hyper[5], hyper[6])
model = Koopman(hidden_layer=2)
print(f'Trainer created.', end = "\n")

# start training
file_name = train_the_model(device, model, loss_function, train_loader, val_loader, hyper, hidden_layer = 2, epochs=100)
plot_learning_curve(file_name)

# get test result
test_the_model(test_loader, model, loss_function, file_name)