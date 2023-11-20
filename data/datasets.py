from torch.utils.data import  DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor

def get_dataloader(dataset, config):
    #split into train and test set
    train_loader_os = DataLoader(dataset[0], batch_size=config.train.batch_size, shuffle=config.train.shuffle, drop_last=config.train.drop_last, pin_memory=False, num_workers=0)
    val_loader_os = DataLoader(dataset[1], batch_size=100, shuffle=False, drop_last=False, num_workers=0)
    test_loader_os = DataLoader(dataset[2], batch_size=config.test.batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    return (train_loader_os, val_loader_os, test_loader_os)


def get_dataset(config):

    #add support for multiple datasets - Fashion MNIST
    if config.dataset.name.lower() == 'mnist': 
        train_dataset = MNIST('./results/data', train=True, download=True, transform=ToTensor())
        test_set = MNIST('./results/data', train=False, download=True, transform=ToTensor())
        train_set, val_set = random_split(train_dataset, [50000, 10000])
    elif config.dataset.name.lower() == 'fashionmnist': 
        train_dataset = FashionMNIST('./results/data', train=True, download=True, transform=ToTensor())
        test_set = FashionMNIST('./results/data', train=False, download=True, transform=ToTensor())
        train_set, val_set = random_split(train_dataset, [50000, 10000])

    return (train_set, val_set, test_set)