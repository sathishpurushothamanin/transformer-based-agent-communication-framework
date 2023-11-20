#Credits
#https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html
#https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
#https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py

import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import editdistance
import numpy as np
from PIL import Image
import  torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure

#display the generated images in a grid
def show(imgs, tag, exp_dir):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(os.path.join(exp_dir, 'test_{}.png'.format(tag)))

def display_codes(data):
  images = data.cpu().detach().numpy()
  batch_words_of_images = dict()
  for i in range(images.shape[0]):
    word = ''
    for j in range(images.shape[1]):
      word = word + str(np.nonzero(images[i][j])[0][0]+1)
    batch_words_of_images[i] = word
  return batch_words_of_images

def show_codes(discrete_latent_code):
    words = list()
    codes = torch.argmax(discrete_latent_code, dim=-1)
    for i in range(codes.size(0)):
        word = ''
        for j in range(codes.size(1)):
            word += str(codes[i][j].tolist())
        words.append(word)
    return words
    
def show_numeric_codes(discrete_latent_code):
    return torch.argmax(discrete_latent_code, dim=-1)


def get_heatmap(model):
    #to do
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    img = Image.open('results/test_100.jpg')
    transform = transforms.Compose([transforms.ToTensor()])
    
    input = transform(img)
    
    #input.unsqueeze_(0)
    
    print(input.size())
    
    preds = model.talk(input)
    
    score, indices = torch.max(preds, 1)
    score.backward()
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    
    plt.figure(figsize=(10,10))
    plt.subplots(1, 2, 1)
    plt.imshow(np.transpose(input.detach().numpy(), (1,2,0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./results/heatmap.png')


def get_editdistance(messages1, messages2):

    return editdistance.distance(messages1, messages2)


def saliency(model, img, config):

    #define transforms to preprocess input image into format expected by model

    #we don't need gradients w.r.t. weights for a trained model
    device = f'cuda:{config.experiment.gpu_id}'
    for param in model.parameters():
        param.requires_grad = False
    
    #set model in eval mode
    model.eval()
    #transoform input PIL image to torch.Tensor and normalize

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input 
    img.unsqueeze_(0)

    img.requires_grad = True
    
    #forward pass to calculate predictions
    q = model.talk(img.to(device))
    q_y = q.view(q.size(0), config.model.latent_dim, config.model.categorical_dim)
    discrete_latent_code1  = F.gumbel_softmax(q_y, config.test.temperature, config.test.hard).view(q.size(0), -1)
        
    #preds = model(input)
    score, indices = torch.max(discrete_latent_code1, -1)
    #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis

    slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)

    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    #plot image and its saleincy map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(img[0].detach().numpy(), (1, 2, 0)))
    #plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(config.test.exp_dir,f'{config.model.name}_image_gradients.png'))

def get_ssim(x_original, reconstructed_images, height, width):
    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)

    sum_ssim = 0

    for i in range(x_original.shape[0]):
        sum_ssim += SSIM(reconstructed_images[i].view(-1, 1, height, width).cpu(), 
                            x_original[i].view(-1, 1, height, width))
    return sum_ssim/x_original.shape[0]