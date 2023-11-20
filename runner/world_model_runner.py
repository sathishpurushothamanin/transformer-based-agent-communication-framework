import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from models.world_model import WorldModel
from data.datasets import get_dataloader, get_dataset
from utils.preprocessing import (snapshot, load_model, )

def train(train_loader_os, world, config, optimizer, epoch):
    train_loss = 0

    device = f'cuda:{config.experiment.gpu_id}'
    world.train().to(device)

    total_accuracy = 0
    accuracy = Accuracy(task="multiclass", num_classes=10)
    for batch_idx, (data, y) in enumerate(train_loader_os):
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        #agent 1 training
        logits = world(data)

        #discover symbols using Gumbul softmax trick
        loss =  F.nll_loss(logits, y)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(data)
        pred = torch.argmax(logits, dim=1)
        train_acc = accuracy(pred.cpu(), y.cpu())
        total_accuracy += train_acc
        if batch_idx % config.train.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Acc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_os.dataset),#
                       100. * batch_idx / len(train_loader_os),
                       loss.item(), train_acc))

 
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader_os.dataset) ))


def train_world_model_runner(config):

    device = f'cuda:{config.experiment.gpu_id}'
    world = WorldModel().to(device)

    print("Start training agents")
    optimizer = optim.Adam(world.parameters(), lr=1e-3)
    
    (train_set, val_set, test_set) = get_dataset(config)
    train_loader_os, val_loader_os, test_loader_os = get_dataloader((train_set, val_set, test_set), config)
    for epoch in range(config.train.epochs):
        train(train_loader_os, world, config, optimizer, epoch)
        
        if epoch >1 and epoch%(config.train.epochs-1) == 0:
            snapshot(world, optimizer, config.save_dir, tag=1)
            

def test_world_model_runner(config):

    device = f'cuda:{config.experiment.gpu_id}'
    world = WorldModel().to(device)

    filename = os.path.join(config.test.exp_dir, "agent_1.pth")
    model_snapshot = torch.load(filename)
    world.load_state_dict(model_snapshot["model"])
    world.eval()

    (train_set, val_set, test_set) = get_dataset(config)
    train_loader_os, val_loader_os, test_loader_os = get_dataloader((train_set, val_set, test_set), config)

    accuracy = Accuracy(task="multiclass", num_classes=10)  
    mprc = MulticlassPrecisionRecallCurve(num_classes=10, thresholds=None)
    for batch_idx, (data, y) in enumerate(test_loader_os):
        data = data.to(device)
        y = y.to(device)
        
        #agent 1 training
        logits = world(data)

        #discover symbols using Gumbul softmax trick
        print('Test Accuracy :', accuracy(torch.argmax(logits, dim=1).cpu(), y.cpu()))
        mprc.update(logits.cpu(), y.cpu())
        break

    fig, ax_ = mprc.plot(score=True)
    fig.savefig(os.path.join(config.test.exp_dir, 'precision_recall_curve.png'))