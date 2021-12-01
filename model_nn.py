import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,in_size=6, out_size=36, hidden_out = 30, hidden_layer = 2):
        super(encoder, self).__init__()
        self.input_layer=nn.Sequential(
            nn.Linear(in_size,hidden_out),
            nn.ReLU()
            )

        layers = []
        for _ in range(hidden_layer):
            layers.append(nn.Linear(hidden_out, hidden_out))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_out, out_size),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class decoder(nn.Module):
    def __init__(self,in_size=36, out_size=6, hidden_out = 30, hidden_layer = 2):
        super(decoder, self).__init__()
        self.input_layer=nn.Sequential(
            nn.Linear(in_size,hidden_out),
            nn.ReLU()
            )

        layers = []
        for _ in range(hidden_layer):
            layers.append(nn.Linear(hidden_out, hidden_out))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_out, out_size),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class linear_system(nn.Module):
    def __init__(self,in_size=33, out_size=30):
        super(linear_system, self).__init__()
        self.layer=nn.Linear(in_size,out_size)

    def forward(self, x):
        x = self.layer(x)
        return x

class Koopman(nn.Module):
    def __init__(self,initial_state=6, lifted_state = 30, hidden_out=30, hidden_layer = 2):
        super(Koopman, self).__init__()
        
        self.en = encoder(initial_state, lifted_state, hidden_out, hidden_layer)
        self.de = decoder(lifted_state, initial_state, hidden_out, hidden_layer)
        self.K = linear_system(lifted_state+3, lifted_state)

    def forward(self,x,u):
        x  = self.en(x)
        x = self.K(torch.cat((x,u)))
        prediction = self.de(x)
        return prediction
    
class Loss(nn.Module):
    def __init__(self,a1, a2, a3, a4, a5, a6, P):
        super(Loss, self).__init__()
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.P = P

    def forward(self, model,x,u):
        x = x.squeeze()
        u = u.squeeze()
        '''
        This is used for pytorch 1.10.0 on Google colab
        en = model.get_submodule("en")
        de = model.get_submodule("de")
        K = model.get_submodule("K")
        '''
        # This is used for pytorch 1.4.0 on Yiping's labtop (newer version also works)
        submodules = []
        for idx, m in enumerate(model.named_children()):
            submodules.append(m)
        en = submodules[0][1]
        de = submodules[1][1]
        K = submodules[2][1]
        #get losses torch.load(saved_model_path)
        # Lx,x = ave(||X(t+i)-decoder(K^i*encoder(Xt))||)
        # Lx,o = ave(||encoder(X(t+i))-K^i*encoder(Xt)||)
        # Lo,x = ave(||Xi-decoder(encoder(Xi))||)
        # Loo = ave(||X(t+i)-decoder(K^i*encoder(Xt))||inf)+ave(||X(t+i)-K^i*encoder(Xt)||)
        mse = nn.MSELoss(reduction='sum')
        Lxx = 0
        Lxo = 0
        Lox = 0
        Loo = 0
        K_i_en_x = en(x[0,:])
        en_x = en(x)
        de_en_x = de(en_x)
        for i in range(self.P):
            K_i_en_x = K(torch.cat((K_i_en_x,u[i,:])))
            pred = de(K_i_en_x)
            Lxx += mse(x[i+1,:],pred)
            Lxo += mse(en_x[i+1,:],K_i_en_x)
            Lox += mse(x[i+1,:],de_en_x[i+1,:])
            Loo += torch.norm(x[i+1,:]-pred,p=float("inf"))+torch.norm(x[i+1,:]-de_en_x[i+1,:],p=float('inf'))
        Lxx /= self.P
        Lxo /= self.P
        Lox /= self.P
        Loo /= self.P

        # get regularization
        L2_en = 0
        for param in en.parameters():
            L2_en += (param ** 2).sum()  
        L2_de = 0
        for param in de.parameters():
            L2_de += (param ** 2).sum()  

        # get the sum
        loss = self.a1*Lxx + self.a2*Lxo + self.a3*Lox + self.a4*Loo + self.a5*L2_en + self.a6*L2_de
        return loss