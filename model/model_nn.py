import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,struct):
        super(encoder, self).__init__()
        #struct = [n,32,64,K,K+n]
        self.struct = struct
        layers = []
        for i in range(len(self.struct)-1):
            layer = nn.Sequential(
                nn.Linear(self.struct[i],self.struct[i+1]),
                nn.ReLU()
                )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        lifted_x = self.layers(x)
        x = torch.cat((x,lifted_x),-1)
        return x

class decoder(nn.Module):
    def __init__(self,struct):
        super(decoder, self).__init__()
        #struct = [K+n,128,64,32,n]
        self.struct = struct
        layers = []
        for i in range(len(self.struct)-1):
            layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.struct[i],self.struct[i+1]),
                )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        return x

class linear_system(nn.Module):
    def __init__(self,lifeted_state):
        super(linear_system, self).__init__()
        self.layer=nn.Linear(lifeted_state+2,lifeted_state,bias=False)

    def forward(self, x,u):
        x = self.layer(torch.cat((x,u),-1))
        return x

class Koopman(nn.Module):
    def __init__(self,en_struct,de_struct):
        super(Koopman, self).__init__()
        
        self.en = encoder(en_struct)
        self.de = decoder(de_struct)
        self.K = linear_system(de_struct[0])

    def forward(self,x,u):
        x  = self.en(x)
        x = self.K(x,u)
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
        # x,u should have 3D structure - (No.trajectory, Time sequence, state)
        '''
        This is used for pytorch 1.10.0 on Google colab
        en = model.get_submodule("en")
        de = model.get_submodule("de")
        K = model.get_submodule("K")
        '''
        # This is used for pytorch 1.4.0 on Yiping's labtop (newer version also works)
        submodules = []
        for idx, m in enumerate(model.children()):
            submodules.append(m)
        en = submodules[0]
        de = submodules[1]
        K = submodules[2]
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
        K_i_en_x = en(x[:,0,:])
        en_x = en(x)
        de_en_x = de(en_x)
        for i in range(self.P):
            K_i_en_x = K(K_i_en_x,u[:,i,:])
            pred = de(K_i_en_x)
            Lxx += mse(x[:,i+1,:],pred)
            Lxo += mse(en_x[:,i+1,:],K_i_en_x)
            Lox += mse(x[:,i+1,:],de_en_x[:,i+1,:])
            Loo += torch.norm(x[:,i+1,:]-pred,p=float("inf"))+torch.norm(x[:,i+1,:]-de_en_x[:,i+1,:],p=float('inf'))
        ave = x.size(0)*self.P
        Lxx /= ave
        Lxo /= ave
        Lox /= ave
        Loo /= ave

        # get regularization
        L2_en = 0
        #for param in en.parameters():
        #    L2_en += (param ** 2).sum()  
        L2_de = 0
        #for param in de.parameters():
        #    L2_de += (param ** 2).sum()  

        # get the sum
        loss = self.a1*Lox + self.a2*Lxx + self.a3*Lxo + self.a4*Loo + self.a5*L2_en + self.a6*L2_de
        return loss

class simple_loss(nn.Module):
    def __init__(self,a1, a2,P):
        super(simple_loss, self).__init__()
        self.a1 = a1
        self.a2 = a2
        self.P = P

    def forward(self, model,x,u):
        mse = nn.MSELoss(reduction='sum')
        acc_loss = 0
        state = x[:,0,:]
        for i in range(self.P):
            state = model(state,u[:,i,:])
            acc_loss += mse(x[:,i+1,:],state)
        acc_loss /= self.P

        # get regularization
        regular = 0
        for param in model.parameters():
            regular += (param ** 2).sum()  

        # get the sum
        loss = self.a1*acc_loss + self.a2*regular
        return loss