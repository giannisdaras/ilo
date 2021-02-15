from ilo_stylegan import LatentOptimizer as StyleGANLatentOptimizer
from ilo_biggan import LatentOptimizer as BigGANLatentOptimizer
from utils import CelebaHQDataset
import yaml
import torchvision
import numpy as np
import torch
import glob
from PIL import Image
import os
import random
import cv2
import glob
import copy
import argparse
import hydra

def sort_name_files(x):
    return int(x.split('_')[-1].split('.')[0])

def save_latent(z, stopping_point, config):
    latent, noises, gen_outs = z
    torch.save(latent, config['saved_noises'][1])
    torch.save(noises, config['saved_noises'][0])
    torch.save(gen_outs[:stopping_point], config['saved_noises'][2])

@hydra.main(config_name='configs/config')
def main(config):
    optimizers = {
        'stylegan': StyleGANLatentOptimizer,
        'biggan': BigGANLatentOptimizer
    }

    datasets = {
        'CelebaHQ': CelebaHQDataset
    }
    model_type = config['model_type']
    config = config[model_type]
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if config['is_dataset']:
        # for now defaults to CelebaHQ
        dataset = datasets[config['dataset_type']](config['input_files'][0])
        config['input_files'] = random.sample(dataset.files, config['num_dataset'])
        print(f'Working with the following files: \n {config["input_files"]}')

    if config['is_video']:
        files = [x for x in glob.iglob(os.path.join(config['input_files'][0], '*' +
                                                    config['files_ext']))]
        files.sort(key=sort_name_files)
        # run first
        config['save_latent'] = True
        config['input_files'] = [files[0]]
        latent_optimizer = optimizers[model_type](config)
        inputs, z, image = latent_optimizer.invert()
        stopping_point = len(config['steps'])
        save_latent(z, stopping_point, config)
        # prepare next
        config['save_latent'] = True
        config['start_layer'] = len(config['steps']) - 1
        config['steps'] = config['per_frame_steps']
        config['max_radius_latent'] = config['max_frame_radius_latent']
        config['max_radius_noises'] = config['max_frame_radius_noises']
        config['max_radius_gen_out'] = config['max_frame_radius_gen_out']
        config['do_project_latent'] = True
        config['do_project_noises'] = True
        config['do_project_gen_out'] = True
        config['restore'] = True

        print('First frame is ready. Running next frames...')
        out_files = []
        for index, file in enumerate(files[1:]):
            print(f'Running frame: {index + 1}/{len(files) - 1}')
            config['input_files'] = [file]
            latent_optimizer.config = config
            latent_optimizer.init_state()

            inputs, z, image = latent_optimizer.invert(reference_vector=image.to(config['device']))

            save_latent(z, stopping_point, config)

            out_file = config['output_files'][0] + file.split('/')[-1]
            out_files.append(out_file)
            torchvision.utils.save_image(
                image,
                out_file,
                nrow=int(image.shape[0] ** 0.5),
                normalize=True)
        out = cv2.VideoWriter(os.path.join(config['output_files'][0], 'frames.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                              config['video_freq'], (1024, 1024))
        images = [cv2.imread(x) for x in out_files]
        for image in images:
             out.write(image)
        out.release()
    else:
        if config['is_sequence']:
            in_files = [x for x in glob.iglob(os.path.join(config['input_files'][0], '*' +
                                                           config['files_ext']))]
            base_names = [x.split('/')[-1] for x in in_files]
            out_files = [os.path.join(config['output_files'][0], x) for x in base_names]
            print('Running for a sequence of images...')
        else:
            in_files = copy.deepcopy(config['input_files'])
            out_files = copy.deepcopy(config['output_files'])
        index = 1
        latent_optimizer = None
        for in_file, out_file in zip(in_files, out_files):
            print(f'Running file {index}/{len(in_files)}')
            config['input_files'] = [in_file]
            config['output_files'] = [out_file]
            if latent_optimizer is None:
               latent_optimizer = optimizers[model_type](config)
            latent_optimizer.config = config
            latent_optimizer.init_state()
            inputs, z, images = latent_optimizer.invert()
            if config['save_latent']:
                save_latent(z, 18, config)
            print(f'Saving file: {out_file}')
            torchvision.utils.save_image(
                images,
                out_file,
                nrow=int(images.shape[0] ** 0.5),
                normalize=True)
            index += 1


if __name__ == "__main__":
    main()
