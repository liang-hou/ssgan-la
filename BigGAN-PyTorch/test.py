''' Test
   This script loads a pretrained net and a weightsfile and test '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses


def testD(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
                
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  D = model.Discriminator(**config).cuda()
  utils.count_parameters(D)
  
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(None, D, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     None,
                     strict=False, load_optim=False)
  
  print('Putting D in eval mode..')
  D.eval()

  class LR(nn.Module):
    def __init__(self, ndim):
      super().__init__()
      self.ndim = ndim
      self.linear = nn.Linear(self.ndim, config['n_classes'])
    
    def forward(self, x):
      return self.linear(x.view(-1, self.ndim))
  
  # config['dataset'] = 'C100'
  # config['n_classes'] = utils.nclass_dict[config['dataset']]
  
  for layer in range(0, len(D.arch['out_channels'])):
    out_channels = D.arch['out_channels'][layer]
    out_resolution = D.arch['resolution'][layer]
    ndim = out_resolution * out_resolution * out_channels
    model = LR(ndim).to(device)
    optimizer = optim.Adam(model.parameters(), 0.05)

    loaders = utils.get_data_loaders(**{**config, 'batch_size': 100,
                                      'start_itr': 0, 'split': 'train'})
  
    model.train()
    for epoch in tqdm(range(50)):
      if epoch == 30 or epoch == 40:
        for param_group in optimizer.param_groups:
          param_group['lr'] /= 10
      if config['pbar'] == 'mine':
        pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
      else:
        pbar = tqdm(loaders[0])
      for i, (x, y) in enumerate(pbar):
        if config['D_fp16']:
          x, y = x.to(device).half(), y.to(device)
        else:
          x, y = x.to(device), y.to(device)
        h = x
        for index, blocklist in enumerate(D.blocks):
          for block in blocklist:
            h = block(h)
          if index == layer: break
        p = model(h)
        loss = F.cross_entropy(p, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    loaders = utils.get_data_loaders(**{**config, 'batch_size': 100,
                                      'start_itr': 0, 'split': 'test'})
    if config['dataset'] == 'TINY':
      config_test = config.copy()
      config_test['dataset'] = 'TINY_val'
      loaders = utils.get_data_loaders(**{**config_test, 'batch_size': 100,
                                      'start_itr': 0, 'split': 'test'})

    acc = 0
    model.eval()
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
    for i, (x, y) in enumerate(pbar):
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      h = x
      for index, blocklist in enumerate(D.blocks):
        for block in blocklist:
          h = block(h)
        if index == layer: break
      p = model(h)
      acc += (p.argmax(1) == y).sum().item()
    print(acc)

def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  # parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  # for weights in range(5000, 50001, 5000):
    # config['load_weights'] = str(weights)
  print(config)
  testD(config)
  
if __name__ == '__main__':    
  main()
