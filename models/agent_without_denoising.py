import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#Transformer-based auto encoder
#some pytorch versions accept this module
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(8, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        inp_x = inp_x.reshape(-1, 8, 8)
        x = x + self.attn(inp_x, inp_x, inp_x)[0].reshape(-1, 64)
        x = x + self.linear(self.layer_norm_2(x))
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# We define a single agent here. An agent observes / creates objects and hears/talks about objects using symbols
class SymbolicAgentWithoutDenoiser (nn.Module):
    
    def __init__(self,
        height,
        width,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        latent_dim,
        categorical_dim,
        dropout=0.0) -> None: # todo: pass architecture parameter here
        super(SymbolicAgentWithoutDenoiser, self).__init__()
        # object encoder/decoder


        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        #self.patch_size = patch_size

        # Layers/Networks
        #self.input_layer_encoder = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.height = height
        self.width = width

        self.input_layer_encoder = nn.Linear(self.height*self.width, embed_dim)
        
        self.transformer_encoder = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head_encoder = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, latent_dim * categorical_dim))

        self.input_layer_decoder = nn.Linear(latent_dim * categorical_dim, embed_dim)
        self.symbol_conditions = nn.Linear(latent_dim * categorical_dim, embed_dim)
        self.transformer_decoder = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head_decoder = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim,self.height*self.width))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        #self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, self.height*self.width))
        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, latent_dim * categorical_dim))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def talk(self, x ):
        x = x.view(-1, self.height*self.width)
        x = x + self.encoder_pos_embedding
        
        
        x = self.input_layer_encoder(x)
        x = self.dropout(x)

        # Apply Transforrmer
        x = self.transformer_encoder(x)

        #cls = x[0]

        q = self.mlp_head_encoder(x)

        return q
    
    def draw(self, z ):
        
        z = z + self.decoder_pos_embedding
        x = self.input_layer_decoder(z)
        x = self.transformer_decoder(x)
        out = self.mlp_head_decoder(x)

        return self.sigmoid(out)

    
    def forward(self, image, temperature, hard):

        q = self.talk(image)
        #get symbols
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        discrete_latent_code  = F.gumbel_softmax(q_y, temperature, hard).reshape(q.size(0), -1)
        images = self.draw(discrete_latent_code.view(q.size(0), -1))
        
        return images, discrete_latent_code
        
def display_model(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
      print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def loss_function(recon_x, x):

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]
        
        #log_ratio = torch.log(qy * categorical_dim + 1e-20)
        #KLD = torch.sum(qy * log_ratio, dim=-1).mean()

        return BCE**2 #convergence better
        #return BCE + KLD