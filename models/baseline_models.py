import torch.nn as nn
import torch.nn.functional as F

#Fully Connected or Dense layer based autoencoder

class LinearAutoEncoder(nn.Module):    
    def __init__(self, height, width, hidden_dim, latent_dim, categorical_dim,):
        super(LinearAutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.obj_encoder = nn.Sequential(
                            nn.Linear(height*width, hidden_dim),
                            nn.ReLU(True),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(True),
                            nn.Linear(hidden_dim, latent_dim*categorical_dim)
                            )
        self.sym_decoder = nn.Sequential(
                            nn.Linear(latent_dim*categorical_dim, hidden_dim),
                            nn.ReLU(True), 
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(True), 
                            nn.Linear(hidden_dim, height*width)
                            )
    def talk(self, image):
        return self.obj_encoder(image.view(-1, self.height*self.width))

    def draw(self, text_seq):
        return self.sym_decoder(text_seq.view(-1, self.latent_dim*self.categorical_dim))
        
    def forward(self, data, temperature, hard):
        q = self.talk(data)
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        discrete_latent_code  = F.gumbel_softmax(q_y, temperature, hard).reshape(q.size(0), -1)
        return self.draw(discrete_latent_code.view(q.size(0), -1)), discrete_latent_code


class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, height, width, hidden_dim, latent_dim, categorical_dim):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        
        self.obj_encoder = nn.Sequential(
                            nn.Conv2d(1, 32, 3, 1),
                            nn.ReLU(True),
                            nn.Conv2d(32, 64, 3, 1),
                            nn.ReLU(True),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.25),
                            nn.Flatten(1),
                            nn.Linear(9216, 128),
                            nn.ReLU(True),
                            nn.Dropout(0.5),
                            nn.Linear(128, hidden_dim),
                            nn.ReLU(True),
                            nn.Linear(hidden_dim, latent_dim*categorical_dim)
                            )
        self.sym_decoder = nn.Sequential(
                            nn.Linear(latent_dim*categorical_dim, hidden_dim),
                            nn.ReLU(True), 
                            nn.Linear(hidden_dim, 128),
                            nn.ReLU(True),
                            nn.Linear(128, 9216),
                            nn.ReLU(True),
                            nn.Linear(9216, self.height*self.width)
                            )
    def talk(self, image):
        X = self.obj_encoder(image)
        return X
    
    def draw(self, text_seq):
        X = self.sym_decoder(text_seq.view(-1, self.latent_dim*self.categorical_dim))
        return X 
    
    def forward(self, data, temperature, hard):
        q = self.talk(data)
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        discrete_latent_code  = F.gumbel_softmax(q_y, temperature, hard).reshape(q.size(0), -1)
        return self.draw(discrete_latent_code.view(q.size(0), -1)), discrete_latent_code
#Generative Adversarial Networks based autoencoder