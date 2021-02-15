import torchvision
import numpy as np
import math
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import lpips
from model import Generator
torch.set_printoptions(precision=5)
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from utils import *


def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


class MappingProxy(nn.Module):
    def __init__(self,gaussian_ft):
        super(MappingProxy,self).__init__()
        self.mean = gaussian_ft["mean"]
        self.std = gaussian_ft["std"]
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.lrelu(self.std * x + self.mean)
        return x

def loss_geocross(latent):
        if latent.size() == (1, 512):
            return 0
        else:
            num_latents  = latent.size()[1]
            X = latent.view(-1, 1, num_latents, 512)
            Y = latent.view(-1, num_latents, 1, 512)
            A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
            B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
            D = 2 * torch.atan2(A, B)
            D = ((D.pow(2) * 512).mean((1, 2)) / 8.).mean()
            return D

class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])


class LatentOptimizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['image_size'][0] != config['image_size'][1]:
            raise Exception('Non-square images are not supported yet.')

        device = config['device']
        self.downsampler_1024_256 = BicubicDownSample(4)
        self.downsampler_1024_image = BicubicDownSample(1024 // config['image_size'][0])
        self.downsampler_image_256 = BicubicDownSample(config['image_size'][0] // 256)

        # Load models and pre-trained weights
        gen = Generator(1024, 512, 8)
        gen.load_state_dict(torch.load(config["ckpt"])["g_ema"], strict=False)
        gen.eval()
        self.gen = gen.to(device)
        self.gen.start_layer = config['start_layer']
        self.gen.end_layer = config['end_layer']
        self.mpl = MappingProxy(torch.load('gaussian_fit.pt'))
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg",
                                            use_gpu=device.startswith("cuda"))
        self.init_state()

    def init_state(self):
        device = self.config['device']
        self.project = self.config["project"]
        self.steps = self.config["steps"]

        self.layer_in = None
        self.best = None
        self.current_step = 0

        transform_lpips = get_transformation(256)
        transform = get_transformation(self.config['image_size'])

        # load images
        original_imgs = []
        for imgfile in self.config['input_files']:
            original_imgs.append(transform(Image.open(imgfile).convert("RGB")))
        self.original_imgs = torch.stack(original_imgs, 0).to(device)

        # save filters
        perc = self.config['observed_percentage'] / 100
        m = int(perc * (3 * self.config['image_size'][0] ** 2))
        self.indices = torch.tensor(np.random.choice(np.arange(1024 * 1024 * 3), m, replace=False))
        self.filters = torch.ones((1024 *  1024 * 3), device=self.config['device']).normal_().unsqueeze(0).to(self.config['device'])
        self.sign_pattern = (torch.rand(1024 * 1024 * 3) >
                            0.5).type(torch.int32).to(self.config['device'])
        self.sign_pattern = 2 * self.sign_pattern - 1
        bs = self.original_imgs.shape[0]
        # initialization
        if self.config['start_layer'] == 0:
            noises_single = self.gen.make_noise(bs)
            self.noises = []
            for noise in noises_single:
                self.noises.append(noise.normal_())
            self.latent_z = torch.randn(
                        (bs, 18, 512),
                        dtype=torch.float,
                        requires_grad=True, device='cuda')
            self.gen_outs = [None]
        if self.config['restore']:
            # restore noises
            self.noises = torch.load(self.config['saved_noises'][0])
            self.latent_z = torch.load(self.config['saved_noises'][1]).to(self.config['device'])
            self.gen_outs = torch.load(self.config['saved_noises'][2])
            self.latent_z.requires_grad = True
        if self.config['start_layer'] != 0 and not self.config['restore']:
            raise NotImplementedError('Please restore vectors or start from the initial layer...')


    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp


    def invert_(self, start_layer, noise_list, steps, index, reference_vector=None):
        learning_rate = self.config['lr'][index]
        print(f"Running round {index + 1} / {len(self.config['steps'])} of ILO.")
        # noise_list containts the indices of nodes that we will be optimizing over
        for i in range(len(self.noises)):
            if i in noise_list:
                self.noises[i].requires_grad = True
            else:
                self.noises[i].requires_grad = False
        with torch.no_grad():
            if start_layer == 0:
                var_list = [self.latent_z] + self.noises
            else:
                self.gen_outs[-1].requires_grad = True
                var_list = [self.latent_z] + self.noises + [self.gen_outs[-1]]
                prev_gen_out = torch.ones(self.gen_outs[-1].shape, device=self.gen_outs[-1].device) * self.gen_outs[-1]
            prev_latent = torch.ones(self.latent_z.shape, device=self.latent_z.device) * self.latent_z
            prev_noises = [torch.ones(noise.shape, device=noise.device) * noise for noise in
                           self.noises]

            # set network that we will be optimizing over
            self.gen.start_layer = start_layer
            self.gen.end_layer = self.config['end_layer']

        optimizer = optim.Adam(var_list, lr=learning_rate)
        ps = SphericalOptimizer([self.latent_z] + self.noises)
        pbar = tqdm(range(steps))
        self.current_step += steps

        # mask the totally black pixels
        curr_shape = self.original_imgs.shape
        mask = torch.ones(curr_shape, device=self.config['device'])
        if self.config['mask_black_pixels']:
            bs, x, y = torch.where(self.original_imgs.sum(dim=1) == -3)
            mask[bs, :, x, y] = 0

        mse_min = np.inf

        mse_loss = 0
        p_loss = 0
        reference_loss = 0
        for i in pbar:
            if self.config['lr_same_pace']:
                total_steps = sum(self.config['steps'])
                t = i / total_steps
            else:
                t = i / steps
            lr = self.get_lr(t, learning_rate)
            optimizer.param_groups[0]["lr"] = lr
            latent_w = self.mpl(self.latent_z)
            img_gen, _ = self.gen([latent_w],
                                  input_is_latent=True,
                                  noise=self.noises,
                                  layer_in=self.gen_outs[-1],)
            batch, channel, height, width = img_gen.shape
            factor = height // 256


            #-                      Calculate loss                           -#
            loss = 0

            if self.config['fast_compress']:
                # TODO: check how to generalize for different sizes...
                real_obsv = partial_circulant_torch(self.original_imgs, self.filters, self.indices,
                                                   self.sign_pattern)
                gen_obsv = partial_circulant_torch(img_gen, self.filters, self.indices,
                                                   self.sign_pattern)
                mse_loss = F.mse_loss(real_obsv, gen_obsv).mean()
                reference_vector = self.original_imgs
                loss += mse_loss
            else:
                # downsample generared images
                downsampled = self.downsampler_1024_image(img_gen)
                # mask
                masked = downsampled * mask
                # compute loss
                diff = torch.abs(masked - self.original_imgs) - self.config['dead_zone_linear_alpha']
                loss += self.config['dead_zone_linear'][index] * torch.max(torch.zeros(diff.shape, device=diff.device), diff).mean()

                mse_loss = F.mse_loss(masked, self.original_imgs)
                loss += self.config['mse'][index] * mse_loss
                if self.config['pe'][index] != 0:
                    if self.config['lpips_method'] == 'mask':
                        p_loss = self.percept(self.downsampler_image_256(masked),
                                              self.downsampler_image_256(self.original_imgs)).mean()
                    elif self.config['lpips_method'] == 'fill':
                        filled = mask * self.original_imgs + (1 - mask) * downsampled
                        p_loss = self.percept(self.downsampler_1024_256(img_gen), self.downsampler_image_256(filled)).mean()
                    else:
                        raise NotImplementdError('LPIPS policy not implemented')

                loss += self.config['pe'][index] * p_loss

                loss += self.config['geocross'] * loss_geocross(self.latent_z[2 * start_layer:])

            if reference_vector is not None:
                reference_loss = F.mse_loss(img_gen, reference_vector)
                loss += self.config['reference_loss'] * reference_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.project:
                ps.step()

            if start_layer != 0 and self.config['do_project_gen_out']:
                deviation = project_onto_l1_ball(self.gen_outs[-1] - prev_gen_out,
                                                 self.config['max_radius_gen_out'][index])
                var_list[-1].data = (prev_gen_out + deviation).data
            if self.config['do_project_latent']:
                deviation = project_onto_l1_ball(self.latent_z - prev_latent,
                                                 self.config['max_radius_latent'][index])
                var_list[0].data = (prev_latent + deviation).data
            if self.config['do_project_noises']:
                deviations = [project_onto_l1_ball(noise - prev_noise,
                                                  self.config['max_radius_noises'][index]) for noise,
                             prev_noise in zip(self.noises, prev_noises)]
                for i, deviation in enumerate(deviations):
                    var_list[i+1].data = (prev_noises[i] + deviation).data

            if (reference_vector is not None) and self.config['save_on_ref'] and reference_loss < mse_min:
                mse_min = reference_loss
                self.best = img_gen.detach().cpu()
            elif mse_loss < mse_min:
                mse_min = mse_loss
                self.best = img_gen.detach().cpu()

            pbar.set_description(
                (
                    f"perceptual: {p_loss:.4f};"
                    f" mse: {mse_loss:.4f};"
                )
            )
            if self.config['save_gif'] and i % self.config['save_every'] == 0:
                torchvision.utils.save_image(
                    img_gen,
                    f'gif_{start_layer}_{i}.png',
                    nrow=int(img_gen.shape[0] ** 0.5),
                    normalize=True)
        # TODO: check what happens when we are in the last layer
        with torch.no_grad():
            latent_w = self.mpl(self.latent_z)
            self.gen.end_layer = self.gen.start_layer
            intermediate_out, _  = self.gen([latent_w],
                                             input_is_latent=True,
                                             noise=self.noises,
                                             layer_in=self.gen_outs[-1],
                                             skip=None)
            self.gen_outs.append(intermediate_out)
            self.gen.end_layer = self.config['end_layer']

    def invert(self, reference_vector=None):
        print('Running with the following config....')
        pretty(self.config)
        for i, steps in enumerate(self.config['steps']):
            begin_from = i + self.config['start_layer']
            if begin_from > self.config['end_layer']:
                raise Exception('Attempting to go after end layer...')
            self.invert_(begin_from, range(5 + 2 * begin_from), int(steps), i, reference_vector)
        return self.original_imgs, (self.latent_z, self.noises, self.gen_outs), self.best
