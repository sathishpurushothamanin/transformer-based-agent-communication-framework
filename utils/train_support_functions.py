import torch
import torch.nn.functional as F

def loss_function(recon_x, x):

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]
        
        #log_ratio = torch.log(qy * categorical_dim + 1e-20)
        #KLD = torch.sum(qy * log_ratio, dim=-1).mean()

        return BCE**2 #convergence better
        #return BCE + KLD
        
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)
def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)
def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)