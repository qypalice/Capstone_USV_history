from trainer import *
from model_nn import Koopman

# create dataset (optional)
x_range = np.array([0.1,0.1,0.1,0.5,0.5,1.5])
u_range = np.array([0.1,0.1,0.1])
SimLength=10
Ntraj = 10
Ts=0.01
path = produce_dataset(x_range,u_range,SimLength,Ntraj,Ts)

# get dataloaders
path = './dataset/x-[0.1 0.1 0.1 0.5 0.5 1.5]_u-[0.1 0.1 0.1]_10x0.1_Ts_0.01'
train_loader, val_loader, test_loader = get_dataloaders(path)

# create trainers
device = "cuda:0" if torch.cuda.is_available() else "cpu"
hyper = [1.0,1.0,0.2,0.000000001,0.0000000001,0.000000001,7]
loss_function = Loss(hyper[0], hyper[1], hyper[2], hyper[3], hyper[4], hyper[5], hyper[6])
model = Koopman(initial_state=6, lifted_state = 10, hidden_out=10, hidden_layer = 2)
trainer = Trainer(device, model, loss_function, train_loader, val_loader, batch_size = 16)
print(f'Trainer created.', end = "\n")

# start training
file_name = train_the_model(trainer, hyper, hidden_layer = 2, epochs=200)
plot_learning_curve(file_name)

# get test result
test_the_model(test_loader, model, loss_function, file_name)