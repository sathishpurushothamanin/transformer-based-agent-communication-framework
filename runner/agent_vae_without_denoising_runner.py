import os
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from models.world_model import WorldModel
from torchmetrics.functional.multimodal import clip_score
import matplotlib as mpl
from models.agent_without_denoising import loss_function, SymbolicAgentWithoutDenoiser
from models.baseline_models import LinearAutoEncoder
from data.datasets import get_dataloader, get_dataset
from utils.postprocessing import (show, saliency,
                                  get_editdistance,
                                  show_codes)

from utils.preprocessing import (snapshot, )
def train(train_loader_os, agent1, agent2, config, optimizer1, optimizer2, epoch):
    
    agent1.train()
    agent2.train()
    train_loss = 0
    ANNEAL_RATE = config.train.anneal_rate
    temp = config.train.temperature
    device = f'cuda:{config.experiment.gpu_id}'

    loss_dict = {'agent1_reconstruction_loss': list(), 'agent2_reconstruction_loss': list(), 'shared_message_loss': list(), 'agent1_shared_reconstruction_loss': list(), 'agent2_shared_reconstruction_loss': list(),}

    for batch_idx, (data, _) in enumerate(train_loader_os):
        
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # Algorithm 1 line 3: sample t uniformally for every example in the batch

        
        #print(noise_added.shape)
        
        images1, discrete_latent_code1 = agent1(data, temp, config.train.hard)

        #discover symbols using Gumbul softmax trick F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]
        loss1 =  loss_function(images1.view(-1, 1, 28, 28), data)
                
        #agent 1 training
        images2, discrete_latent_code2 = agent2(data, temp, config.train.hard)
 
        #discover symbols using Gumbul softmax trick
        loss2 =  loss_function(images2.view(-1, 1, 28, 28), data)
        
        #combined loss - find aggreement between two agents symbols
        #if two agents find incompatible codes loss will be higher

        images1 = agent2.draw(discrete_latent_code1.view(config.train.batch_size, -1))
        images2 = agent1.draw(discrete_latent_code2.view(config.train.batch_size, -1))
        
        loss3 = loss_function(images1.view(-1, 1, 28, 28), data)
        loss4 = loss_function(images2.view(-1, 1, 28, 28), data)

        loss5 = F.mse_loss(discrete_latent_code1, discrete_latent_code2)

        if config.model.communication_pathway == 1:
            loss = loss1 + loss2 + loss3 + loss4
        elif config.model.communication_pathway == 2:
            loss = loss1 + loss2 + loss5
        elif config.model.communication_pathway == 3:
            loss = loss1 + loss2 + loss3 + loss4 + loss5
        else:
            loss = loss1 + loss2

        loss_dict['agent1_reconstruction_loss'].append(loss1.item())
        loss_dict['agent2_reconstruction_loss'].append(loss2.item())
        loss_dict['shared_message_loss'].append(loss5.item())
        loss_dict['agent1_shared_reconstruction_loss'].append(loss4.item())
        loss_dict['agent2_shared_reconstruction_loss'].append(loss3.item())
                        
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        train_loss += loss.item() * len(data)
        #train_loss += loss3.item() * len(data)
        
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-config.train.anneal_rate * batch_idx), config.train.temp_min)

        if batch_idx % config.train.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_os.dataset),
                       100. * batch_idx / len(train_loader_os),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader_os.dataset)))
    
    return loss_dict


def val_runner(val_loader_os, agent1, agent2, config, epoch):

    agent1.eval()
    agent2.eval()

    device = f'cuda:{config.experiment.gpu_id}'
    
    for (data, _) in val_loader_os:
        val_data = data.to(device)
        break
    
    images1, discrete_latent_code1 = agent1(val_data, config.test.temperature, config.test.hard)
    images2, discrete_latent_code2 = agent2(val_data, config.test.temperature, config.test.hard)

    set1 = show_codes(discrete_latent_code1.view(config.test.batch_size, config.model.latent_dim, config.model.categorical_dim))
    set2 = show_codes(discrete_latent_code2.view(config.test.batch_size, config.model.latent_dim, config.model.categorical_dim))

    freq = dict()
    for i in range(config.test.batch_size):
        dist = get_editdistance(set1[i], set2[i])
        if dist not in freq:
            freq[dist]  = 1
        else:
            freq[dist] += 1
    
    print(epoch, freq)


def generate(test_loader_os, agent1, agent2, config):

    device = f'cuda:{config.experiment.gpu_id}'
    for (data, y) in test_loader_os:
        val_data = data.to(device)
        break
    
    images1, discrete_latent_code1 = agent1(val_data, config.test.temperature, config.test.hard)
    images2, discrete_latent_code2 = agent2(val_data, config.test.temperature, config.test.hard)

    set1 = show_codes(discrete_latent_code1.view(config.test.batch_size, config.model.latent_dim, config.model.categorical_dim))
    set2 = show_codes(discrete_latent_code2.view(config.test.batch_size, config.model.latent_dim, config.model.categorical_dim))

    data_list = list()

    for i in range(images1.size(0)):
        if config.train.agreement.lower() == 'editdistance':
            if get_editdistance(set1[i], set2[i]) <= 3:
                data_list.append((images1[i].view(-1, config.dataset.height, config.dataset.width).cpu(), y[i].item()))
        if config.train.agreement.lower() == 'clipiqa':
            score = clip_score(images1[i], text ='digit ' + str(y[i]))
            print('Clip Score is ', score.detach().item())
            if  score.detach().item() > 80:
                data_list.append((images1[i].view(-1, config.dataset.height, config.dataset.width).cpu(), y[i].item()))
    
    print('Number of digits discovered by agreement ', len(data_list))

    return data_list

def vae_without_denoising_runner(config):

    config.device = f'cuda:{config.experiment.gpu_id}'
    kwargs={
            "height": config.dataset.height,
            "width": config.dataset.width,
            "embed_dim": config.model.embed_dim,
            "hidden_dim": config.model.hidden_dim,
            "num_heads": config.model.num_heads,
            "num_layers": config.model.num_layers,
            "num_channels": config.model.num_channels,
            "latent_dim": config.model.latent_dim,
            "categorical_dim": config.model.categorical_dim,
            "dropout": config.model.dropout,
        }
    device = f'cuda:{config.experiment.gpu_id}'
    agent1 = SymbolicAgentWithoutDenoiser(**kwargs).to(device)
    agent2 = SymbolicAgentWithoutDenoiser(**kwargs).to(device)
    print("Start training agents".format(config.model.latent_dim))
    optimizer1 = optim.Adam(agent1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(agent2.parameters(), lr=1e-3)
    (train_set, val_set, test_set) = get_dataset(config)
    train_loader_os, val_loader_os, test_loader_os = get_dataloader((train_set, val_set, test_set), config)

    loss_dict = {'agent1_reconstruction_loss': list(), 'agent2_reconstruction_loss': list(), 'shared_message_loss': list(), 'agent1_shared_reconstruction_loss': list(), 'agent2_shared_reconstruction_loss': list(),}

    for epoch in range(config.train.epochs):
        loss_per_epoch = train(train_loader_os, agent1, agent2, config, optimizer1, optimizer2, epoch)
        loss_dict['agent1_reconstruction_loss'].append(np.mean(loss_per_epoch['agent1_reconstruction_loss']))
        loss_dict['agent2_reconstruction_loss'].append(np.mean(loss_per_epoch['agent2_reconstruction_loss']))
        loss_dict['shared_message_loss'].append(np.mean(loss_per_epoch['shared_message_loss']))
        loss_dict['agent1_shared_reconstruction_loss'].append(np.mean(loss_per_epoch['agent1_shared_reconstruction_loss']))
        loss_dict['agent2_shared_reconstruction_loss'].append(np.mean(loss_per_epoch['agent2_shared_reconstruction_loss']))
        val_runner(val_loader_os, agent1, agent2, config, epoch)
        if epoch >1 and epoch%(config.train.epochs-1) == 0:
            snapshot(agent1, optimizer1, config.save_dir, tag=1)
            snapshot(agent2, optimizer2, config.save_dir, tag=2)

    pickle.dump(loss_dict, open(os.path.join(config.test.exp_dir, 'loss_dict.p'), 'wb'))

def vae_without_denoising_test_runner(config):

    kwargs={
        "height": config.dataset.height,
        "width": config.dataset.width,
        "embed_dim": config.model.embed_dim,
        "hidden_dim": config.model.hidden_dim,
        "num_heads": config.model.num_heads,
        "num_layers": config.model.num_layers,
        "num_channels": config.model.num_channels,
        "latent_dim": config.model.latent_dim,
        "categorical_dim": config.model.categorical_dim,
        "dropout": config.model.dropout,
    }
    device = f'cuda:{config.experiment.gpu_id}'
    agent1 = SymbolicAgentWithoutDenoiser(**kwargs).to(device)
    agent2 = SymbolicAgentWithoutDenoiser(**kwargs).to(device)
    filename = os.path.join(config.test.exp_dir, "agent_1.pth")
    model_snapshot = torch.load(filename)
    agent1.load_state_dict(model_snapshot["model"])
    agent1.eval()
    
    filename = os.path.join(config.test.exp_dir, "agent_2.pth")
    model_snapshot = torch.load(filename)
    agent2.load_state_dict(model_snapshot["model"])
    agent2.eval()
    
    #display_model(agent)

    sample_size=100
    config.dataset.allowed_digits = 9
    (train_set, val_set, test_set) = get_dataset(config)
    train_loader_os, val_loader_os, test_loader_os = get_dataloader((train_set, val_set, test_set), config)
    #get symbols
    
    image_dict = dict()

    test_list = list()

    for x, y in test_loader_os:
      x_original = x
      y_original = y
      break

    saliency(agent1, x_original[0], config)  

    show(torchvision.utils.make_grid(x_original.view(-1, 1, config.dataset.height, config.dataset.width)[:20]), 0, config.test.exp_dir)
    #auto encoder 1 generated message
    q = agent1.talk(x_original[:sample_size].to(device))
    q_y = q.view(q.size(0), config.model.latent_dim, config.model.categorical_dim)
    discrete_latent_code1  = F.gumbel_softmax(q_y, config.test.temperature, config.test.hard).reshape(q.size(0), -1)
    
    #auto encoder 2 generated message
    q = agent2.talk(x_original[:sample_size].to(device))
    q_y = q.view(q.size(0), config.model.latent_dim, config.model.categorical_dim)
    discrete_latent_code2  = F.gumbel_softmax(q_y, config.test.temperature, config.test.hard).reshape(q.size(0), -1)
    
    #agent 1 creates object using auto encoder 2 message
    images1 = agent1.draw(discrete_latent_code1.view(q.size(0), -1))


    device = f'cuda:{config.experiment.gpu_id}'
    world = WorldModel().to(device)

    filename = "results/world_model/agent_1.pth"
    model_snapshot = torch.load(filename)
    world.load_state_dict(model_snapshot["model"])
    world.eval()

    mpl.rcParams['axes.titlesize'] = 24
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 'Large'
    mpl.rcParams['figure.figsize'] = (6,6)
    mpl.rcParams['figure.dpi'] = 300

    
    accuracy = Accuracy(task="multiclass", num_classes=10)  
    mprc = MulticlassPrecisionRecallCurve(num_classes=10, thresholds=None)

    data = images1.view(-1, 1, config.dataset.height, config.dataset.width).to(device)
    y = y_original.to(device)
    
    #agent 1 training
    logits = world(data)

    metrics = dict()
    metrics['agent1'] = dict()
    metrics['agent2'] = dict()

    #discover symbols using Gumbul softmax trick
    print('Test Accuracy :', accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu()))

    metrics['agent1']['reconstruction_accuracy'] = accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu())


    
    




    mprc.update(logits.cpu(), y.cpu())

    fig, ax_ = mprc.plot(score=True)
    fig.savefig(os.path.join(config.test.exp_dir, 'precision_recall_curve.png'))



    imgs = torchvision.utils.make_grid(images1.view(-1, 1, config.dataset.height, config.dataset.width)[:20])
    show(imgs, 11, config.test.exp_dir)
    
    #agent 2 creates object using auto encoder 1 message
    images2 = agent2.draw(discrete_latent_code2.view(q.size(0), -1))
    
    data = images2.view(-1, 1, config.dataset.height, config.dataset.width).to(device)
    y = y_original.to(device)
    
    #agent 1 training
    logits = world(data)

    #discover symbols using Gumbul softmax trick
    
    metrics['agent2']['reconstruction_accuracy'] = accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu())

    
    imgs = torchvision.utils.make_grid(images2.view(-1, 1, config.dataset.height, config.dataset.width)[:20])
    show(imgs, 21, config.test.exp_dir)


    set1 = show_codes(discrete_latent_code1.view(config.test.batch_size, config.model.latent_dim, config.model.categorical_dim))
    set2 = show_codes(discrete_latent_code2.view(config.test.batch_size, config.model.latent_dim, config.model.categorical_dim))

    freq = dict()
    for i in range(config.test.batch_size):
        dist = get_editdistance(set1[i], set2[i])
        if dist not in freq:
            freq[dist]  = 1
        else:
            freq[dist] += 1


    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)
    PSNR = PeakSignalNoiseRatio()
    print('Structural Similarity Index Measure ')
    

    sum_ssim_agent1 = 0
    sum_ssim_agent2 = 0
    sum_psnr_agent1 = 0
    sum_psnr_agent2 = 0
    for i in range(100):
        sum_ssim_agent1 += SSIM(images1[i].view(-1, 1, config.dataset.height, config.dataset.width).cpu(), 
                            x_original[i].view(-1, 1, config.dataset.height, config.dataset.width))
        sum_ssim_agent2 += SSIM(images2[i].view(-1, 1, config.dataset.height, config.dataset.width).cpu(), 
                            x_original[i].view(-1, 1, config.dataset.height, config.dataset.width))
        sum_psnr_agent1 += PSNR(images1[i].view(-1, 1, config.dataset.height, config.dataset.width).cpu(), 
                            x_original[i].view(-1, 1, config.dataset.height, config.dataset.width))
        sum_psnr_agent2 += PSNR(images2[i].view(-1, 1, config.dataset.height, config.dataset.width).cpu(), 
                            x_original[i].view(-1, 1, config.dataset.height, config.dataset.width))
    
    metrics['agent1']['ssim'] = sum_ssim_agent1/config.test.batch_size
    metrics['agent2']['ssim'] = sum_ssim_agent2/config.test.batch_size
    
    print('Average SSIM for agent 1: ', sum_ssim_agent1/config.test.batch_size)
    print('Average SSIM for agent 2: ', sum_ssim_agent2/config.test.batch_size)
    
    print('Average PSNR for agent 1: ', sum_psnr_agent1/config.test.batch_size)
    print('Average PSNR for agent 2: ', sum_psnr_agent2/config.test.batch_size)

    print(freq)

    metrics['shared_agent1'] = dict()
    metrics['shared_agent2'] = dict()

    agent1_shared_reconstructed_images = agent1.draw(discrete_latent_code2.view(q.size(0), -1))
    
    data = agent1_shared_reconstructed_images.view(-1, 1, config.dataset.height, config.dataset.width).to(device)
    y = y_original.to(device)
    
    #agent 1 training
    logits = world(data)

    #discover symbols using Gumbul softmax trick
    
    metrics['shared_agent1']['reconstruction_accuracy'] = accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu())

    agent2_shared_reconstructed_images = agent2.draw(discrete_latent_code1.view(q.size(0), -1))
    
    data = agent2_shared_reconstructed_images.view(-1, 1, config.dataset.height, config.dataset.width).to(device)
    y = y_original.to(device)
    
    #agent 1 training
    logits = world(data)

    #discover symbols using Gumbul softmax trick
    
    metrics['shared_agent2']['reconstruction_accuracy'] = accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu())
    metrics['shared_agent1']['ssim'] = get_ssim(x_original, agent1_shared_reconstructed_images, config.dataset.height, config.dataset.width)
    metrics['shared_agent2']['ssim'] = get_ssim(x_original, agent2_shared_reconstructed_images, config.dataset.height, config.dataset.width)

    metrics['edit_distance'] = freq
    
    pickle.dump(metrics, open(os.path.join(config.test.exp_dir, f'metrics_{config.model.latent_dim}.p'), 'wb'))

    print(metrics)


def get_ssim(x_original, reconstructed_images, height, width):
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)

    sum_ssim = 0

    for i in range(x_original.shape[0]):
        sum_ssim += SSIM(reconstructed_images[i].view(-1, 1, height, width).cpu(), 
                            x_original[i].view(-1, 1, height, width))
    return sum_ssim/x_original.shape[0]