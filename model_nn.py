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

    def forward(self, x,u):
        x  = self.en(x)
        x = self.K(torch.cat((x,u),2))
        prediction = self.de(x)
        return prediction