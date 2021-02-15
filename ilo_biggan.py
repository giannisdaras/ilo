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
import yaml
import logging
from biggan import Generator
from PIL import Image
from torchvision import transforms
import lpips
from utils import *


def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])


def load_weights(G, D, state_dict, weights_root, experiment_name,
                 name_suffix=None, G_ema=None, strict=True,
                 load_optim=True,
                 map_location='cuda'):
  root = '/'.join([weights_root, experiment_name])
  if name_suffix:
    print('Loading %s weights from %s...' % (name_suffix, root))
  else:
    print('Loading weights from %s...' % root)
  if G is not None:
    G.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix])), map_location=map_location),
      strict=strict)
    if load_optim:
      G.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))), map_location=map_location)
  if D is not None:
    D.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])), map_location=map_location),
      strict=strict)
    if load_optim:
      D.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))), map_location=map_location)
  # Load state dict
  for item in state_dict:
    try:
        state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])), map_location=map_location)[item]
    except:
        print('Warning: {} not found in state dict'.format(item))
  if G_ema is not None:
    G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix])), map_location=map_location),
      strict=strict)


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False, z_var=1.0, target=None, range=None):

  dtype = torch.float16 if fp16 else torch.float32
  z = torch.empty((G_batch_size, dim_z),
                  device=device,
                  dtype=dtype,
                  requires_grad=False).normal_(0, math.sqrt(z_var))
  if range is not None:
      z.clamp_(-range, range)

  y = torch.empty(G_batch_size, dtype=torch.int64,
                  requires_grad=False, device=device).random_(nclasses)
  if target is not None:
      y.fill_(target)

  return z, y


class BigGAN:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(**self.config).to(self.config['device'])

    def load_pretrained(self):
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                        'best_IS': 0, 'best_FID': 999999, 'config': self.config}

        if self.config['ema']:
            field_a = None
            field_b = self.generator
        else:
            field_a = self.generator
            field_b = None
        load_weights(field_a,
                     None,
                     state_dict,
                     '.',
                     '138k',
                     '',
                     field_b,
                     strict=False,
                     load_optim=False,
                     map_location=self.config['device'])

        logging.log(logging.INFO, 'Weights loaded...')
        self.generator.to(self.config['device']).eval()
        logging.log(logging.INFO, 'Generator on eval mode')



def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])




class LatentOptimizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['image_size'][0] != config['image_size'][1]:
            raise Exception('Non-square images are not supported yet.')
        self.reconstruction = config["reconstruction"]
        self.steps = config["steps"]
        self.lr = config['lr']
        # self.mse_record = []

        transform = get_transformation(config['image_size'][0])
        images = []

        for imgfile in config['input_files']:
            images.append(transform(Image.open(imgfile).convert("RGB")))

        self.images = torch.stack(images, 0).to(config['device'])
        self.downsampler_256_image = BicubicDownSample(256 // config['image_size'][0])

        biggan = BigGAN(config)
        biggan.load_pretrained()
        self.generator = biggan.generator
        self.generator.eval()
        self.generator.to(config['device'])

        (self.z, y) = prepare_z_y(
                        self.images.shape[0],
                        self.generator.dim_z,
                        config["n_classes"],
                        device=config["device"],
                        fp16=config["G_fp16"],
                        z_var=config["z_var"],
                        target=config["target"],
                        range=config["range"])
        self.y = self.generator.shared(y)

        self.z.requires_grad = True

        self.perceptual_loss = lpips.PerceptualLoss(model="net-lin", net="vgg",
                                                    use_gpu=config['device'].startswith("cuda"))


    def get_lr(self, step_percentage, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - step_percentage) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, step_percentage / rampup)
        return initial_lr * lr_ramp


    def _invert(self, steps):
        optimizer = torch.optim.Adam([self.z], lr=self.lr)

        if self.reconstruction == 'invert':
            mask = np.ones(self.config['image_size'])
        elif self.reconstruction == 'inpaint':
            mask = torch.ones(self.config['image_size'], device=self.config['device'])
            _, _, x, y = torch.where(self.images == -1)
            mask[x, y] = 0
        pbar = tqdm(range(steps))
        mse_min = np.inf



        for current_step in pbar:
            img_gen, _ = self.generator(self.z, self.y)
            lr = self.get_lr(current_step / steps, self.lr)
            optimizer.param_groups[0]["lr"] = lr

            if self.reconstruction == 'invert':
                # TODO: downsample here
                mse_loss = F.mse_loss(img_gen, self.images)
                perceptual_loss = self.perceptual_loss(img_gen, self.images).sum()
                loss = mse_loss + perceptual_loss
            elif self.reconstruction == 'inpaint':
                mse_loss = F.mse_loss(img_gen * mask, self.images)
                if self.config['lpips_method'] == 'mask':
                    perceptual_loss = self.perceptual_loss(img_gen * mask, self.images).sum()
                elif self.config['lpips_method'] == 'fill':
                    perceptual_loss = self.perceptual_loss(img_gen, self.images * mask + (1 - mask) * img_gen).sum()
                else:
                    raise NotImplementedError('LPIPS configuration not implemented.')
            loss = self.config['mse'] * mse_loss + self.config['pe'] * perceptual_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # self.mse_record.append(mse_loss.item().detach().cpu())

            optimizer.step()

            if mse_loss < mse_min:
                mse_min = mse_loss
                self.best = img_gen

            pbar.set_description(
                (
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

    def invert(self):
        for i, steps in enumerate(self.steps.split(',')):
            self._invert(int(steps))
            # get input for next step
            self.generator.end_layer = self.generator.start_layer + 1
            h, self.y = self.generator(self.z, self.y)

            self.z = h.detach()
            self.z.requires_grad = True

            # restore end layer
            self.generator.end_layer = len(self.generator.blocks)
            # move one step
            self.generator.start_layer += 1
        return self.z, self.best
