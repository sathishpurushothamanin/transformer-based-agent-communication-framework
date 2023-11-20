import os
import torch


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]0495898089
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

def snapshot(model, optimizer, save_dir, tag=None):
  model_snapshot = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tag": tag
  }


  torch.save(model_snapshot,
             os.path.join(save_dir, "agent_{}.pth".format(tag)))


def load_model(model, file_name):
  model_snapshot = torch.load(file_name)
  model.load_state_dict(model_snapshot["model"])
  return model

