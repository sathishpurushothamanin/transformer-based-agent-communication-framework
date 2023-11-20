import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np

from models.baseline_models import LinearAutoEncoder, ConvolutionalAutoEncoder
from data.datasets import get_dataloader, get_dataset
from utils.postprocessing import (show, saliency,
                                  get_editdistance,
                                  show_codes)
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from models.world_model import WorldModel
from utils.preprocessing import (snapshot, load_model, )
from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)

def train(train_loader_os, agent1, agent2, config, optimizer1, optimizer2, epoch):
    
    agent1.train()
    agent2.train()
    train_loss = 0
    ANNEAL_RATE = config.train.anneal_rate
    temp = config.train.temperature
    device = f'cuda:{config.experiment.gpu_id}'
    
    freq_list = list()
    entropy_list = list()
    for batch_idx, (data, _) in enumerate(train_loader_os):
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        #agent 1 training
        images1, discrete_latent_code1 = agent1(data, temp, config.train.hard)


        #discover symbols using Gumbul softmax trick
        loss1 =  F.mse_loss(images1.view(-1, 1, 28, 28), data)

        #agent 1 training
        images2, discrete_latent_code2 = agent2(data, temp, config.train.hard)

        #discover symbols using Gumbul softmax trick
        loss2 =  F.mse_loss(images2.view(-1, 1, 28, 28), data)
        
        images3 = agent2.draw(discrete_latent_code1.view(config.train.batch_size, -1))
        images4 = agent1.draw(discrete_latent_code2.view(config.train.batch_size, -1))
        
        loss3 = F.mse_loss(images3.view(-1, 1, 28, 28), data)
        loss4 = F.mse_loss(images4.view(-1, 1, 28, 28), data)
        loss5 = F.mse_loss(discrete_latent_code1, discrete_latent_code2)
        loss = loss1 + loss2 + loss3 + loss4 + loss5
                
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        train_loss += loss.item() * len(data)
        #loss.test
        
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-config.train.anneal_rate * batch_idx), config.train.temp_min)
        if batch_idx % config.train.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_os.dataset),#
                       100. * batch_idx / len(train_loader_os),
                       loss.item()))

 
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader_os.dataset)))


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

def train_runner(config):

    device = f'cuda:{config.experiment.gpu_id}'
    kwargs={
            "height": config.dataset.height,
            "width": config.dataset.width,
            "hidden_dim": config.model.hidden_dim,
            "latent_dim": config.model.latent_dim,
            "categorical_dim": config.model.categorical_dim
        }
    device = f'cuda:{config.experiment.gpu_id}'
    agent1 = eval(config.model.name)(**kwargs).to(device)
    agent2 = eval(config.model.name)(**kwargs).to(device)
    print("Start training agents".format(config.model.latent_dim))
    optimizer1 = optim.Adam(agent1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(agent2.parameters(), lr=1e-3)
    (train_set, val_set, test_set) = get_dataset(config)
    train_loader_os, val_loader_os, test_loader_os = get_dataloader((train_set, val_set, test_set), config)
    for epoch in range(config.train.epochs):
        train(train_loader_os, agent1, agent2, config, optimizer1, optimizer2, epoch)
        val_runner(val_loader_os, agent1, agent2, config, epoch)
        if epoch >1 and epoch%(config.train.epochs-1) == 0:
            snapshot(agent1, optimizer1, config.save_dir, tag=1)
            snapshot(agent2, optimizer2, config.save_dir, tag=2)

def test_runner(config):

    kwargs={
        "height": config.dataset.height,
        "width": config.dataset.width,
        "hidden_dim": config.model.hidden_dim,
        "latent_dim": config.model.latent_dim,
        "categorical_dim": config.model.categorical_dim
    }
    device = f'cuda:{config.experiment.gpu_id}'
    agent1 = eval(config.model.name)(**kwargs).to(device)
    agent2 = eval(config.model.name)(**kwargs).to(device)

    filename = os.path.join(config.test.exp_dir, "agent_1.pth")
    model_snapshot = torch.load(filename)
    agent1.load_state_dict(model_snapshot["model"])
    agent1.eval()
    
    filename = os.path.join(config.test.exp_dir, "agent_2.pth")
    model_snapshot = torch.load(filename)
    agent2.load_state_dict(model_snapshot["model"])
    agent2.eval()

    sample_size=100
    (train_set, val_set, test_set) = get_dataset(config)
    train_loader_os, val_loader_os, test_loader_os = get_dataloader((train_set, val_set, test_set), config)
    #get symbols
    
    image_dict = dict()
    for x, y in test_loader_os:
      x_original = x
      y_original = y
      break
    
    saliency(agent1, x_original[0], config) 

    y_list = y_original.tolist()
    
    for i, code in enumerate(range(10)):
        image_dict[code] = list()
        
    for i, code in enumerate(y_list):
        image_dict[code].append(i)
    
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
    images1 = agent1.draw(discrete_latent_code2.view(q.size(0), -1))
    
    device = f'cuda:{config.experiment.gpu_id}'
    world = WorldModel().to(device)

    filename = "results/world_model/agent_1.pth"
    model_snapshot = torch.load(filename)
    world.load_state_dict(model_snapshot["model"])
    world.eval()

    
    accuracy = Accuracy(task="multiclass", num_classes=10)  
    mprc = MulticlassPrecisionRecallCurve(num_classes=10, thresholds=None)

    data = images1.view(-1, 1, config.dataset.height, config.dataset.width).to(device)
    y = y_original.to(device)
    
    #agent 1 training
    logits = world(data)

    #discover symbols using Gumbul softmax trick
    print('Test Accuracy :', accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu()))
    mprc.update(logits.cpu(), y.cpu())

    fig, ax_ = mprc.plot(score=True)
    fig.savefig(os.path.join(config.test.exp_dir, 'precision_recall_curve.png'))

    imgs = torchvision.utils.make_grid(images1.view(-1, 1, config.dataset.height, config.dataset.width)[:20])
    show(imgs, 12, config.test.exp_dir)
    
    #agent 2 creates object using auto encoder 1 message
    images2 = agent2.draw(discrete_latent_code1.view(q.size(0), -1))
    
    imgs = torchvision.utils.make_grid(images2.view(-1, 1, config.dataset.height, config.dataset.width)[:20])
    show(imgs, 21, config.test.exp_dir)
    
    
    print('Agent 1 message ')
    set1 = show_codes(discrete_latent_code1.view(q.size(0), config.model.latent_dim, config.model.categorical_dim))
    print('Agent 2 message ')
    set2 = show_codes(discrete_latent_code2.view(q.size(0), config.model.latent_dim, config.model.categorical_dim))

    freq = dict()
    for i in range(sample_size):
        dist = get_editdistance(set1[i], set2[i])
        if dist not in freq:
            freq[dist]  = 1
        else:
            freq[dist] += 1
    
    print('How far are the two sets of codes ', freq)
    
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
    
    print('Average SSIM for agent 1: ', sum_ssim_agent1/config.test.batch_size)
    print('Average SSIM for agent 2: ', sum_ssim_agent2/config.test.batch_size)
    
    print('Average PSNR for agent 1: ', sum_psnr_agent1/config.test.batch_size)
    print('Average PSNR for agent 2: ', sum_psnr_agent2/config.test.batch_size)
            
    #agent 1 creates object using auto encoder 1 message
    images = agent1.draw(discrete_latent_code1.view(q.size(0), -1))
    
    imgs = torchvision.utils.make_grid(images.view(-1, 1, config.dataset.height, config.dataset.width)[:20])
    show(imgs, 11, config.test.exp_dir)
    
    #agent 2 creates object using auto encoder 2 message
    images = agent2.draw(discrete_latent_code2.view(q.size(0), -1))
    
    imgs = torchvision.utils.make_grid(images.view(-1, 1, config.dataset.height, config.dataset.width)[:20])
    show(imgs, 22, config.test.exp_dir)
    