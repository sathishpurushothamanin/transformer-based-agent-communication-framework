import os
import numpy as np
import yaml
import torch
from utils.arg_parser import parse_arguments, get_config, edict2dict
from runner.general_runner import train_runner, test_runner
from runner.agent_vae_without_denoising_runner import vae_without_denoising_runner, vae_without_denoising_test_runner
from runner.world_model_runner import (train_world_model_runner,
                                       test_world_model_runner)
seed = 12

#numpy random seed
np.random.seed(seed)

#pytorch random seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_arguments()
    config = get_config(args.config_file)
    default_exp_dir = config.test.exp_dir
    if args.test:
        eval(config.test.runner)(config)
    elif args.random:
        for seed in range(10):
            config.experiment.random_seed = seed
            for pathways in range(4):
                config.model.communication_pathway = pathways
                if not os.path.exists(os.path.join(default_exp_dir, f'{seed}_{pathways}')):
                    os.mkdir(os.path.join(default_exp_dir, f'{seed}_{pathways}'))

                config.test.exp_dir = os.path.join(default_exp_dir, f'{seed}_{pathways}')
                config.save_dir = config.test.exp_dir
                save_name = os.path.join(config.test.exp_dir, 'config.yaml')
                yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
                eval(config.train.runner)(config)
                eval(config.test.runner)(config)
    else:
        eval(config.train.runner)(config)
