import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import cv2
from align_face import align_face
import torch
from utils import *
import sys
import random
import hydra

def get_arrays(config):
    images = []
    for file in config['input_files']:
        img = np.array(Image.open(file).convert('RGB'))
        img = cv2.resize(img, tuple(config['image_size']))
        images.append(img)
    return images

def interactive_mask_images(config, images):
    new_images = []
    for image in images:
        mask = np.ones(config['image_size'], dtype=np.uint8)
        new_image = image
        for i in range(config['num_rois']):
            r = cv2.selectROI(new_image[:, :, ::-1])
            mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 0
            new_image = (new_image.transpose(2, 0, 1) * mask).transpose(1, 2, 0)
        new_images.append(new_image)
    return new_images

def mask_image(config, images):
    img = images[0]
    box = config['bounding_box']
    y_0, y_1, x_0, x_1 = [x for x in box['horizontal'] + box['vertical']]
    mask = np.ones(config['image_size'], dtype=np.uint8)
    mask[x_0:x_1, y_0:y_1] = 0
    img = (img.transpose(2, 0, 1) * mask).transpose(1, 2, 0)
    return [img]

def mash_images(config, images):
    img_1, img_2 = images
    img = np.ones(img_1.shape, dtype=np.uint8)
    img[0:512, :] = img_1[0:512, :]
    img[512:1024, :] = img_2[512:1024, :]
    return [img]


def blend_images(config, images):
    img_1, img_2 = images
    blend = config['blend']
    return [(blend * img_1 + (1 - blend) * img_2).astype(np.uint8)]


def remove_pixels(config, images):
    new_images = []
    total_pixels = config['image_size'][0] * config['image_size'][1]
    for index, image in enumerate(images):
        for perc in config['observed_percentage']:
            perc = perc / 100
            mask = np.random.binomial(1, perc, config['image_size'])
            new_images.append((image.transpose(2, 0, 1) * mask).transpose(1, 2, 0).astype(np.uint8))
    return new_images


def save_file(image_name, img):
    Image.fromarray(img).save(image_name)


def align_images(config):
    for img_path in config['input_files']:
        new_img = align_face(img_path)
        new_img.save(img_path)

def collect_video_frames(config):
    cap = cv2.VideoCapture(config['input_files'][0])
    base_path = config['output_dir']
    counter = 0
    images = []
    files = []
    while 1:
        ret, frame = cap.read()
        if ret == 0:
            break
        if counter % config['sampling_freq'] == 0:
            image_name = f'{base_path}/frame_{counter}.png'
            cv2.imwrite(image_name, frame)
            frame = align_face(image_name)
            frame.save(image_name)
            images.append(cv2.imread(image_name))
            files.append(image_name)
        counter += 1
    config['input_files'] = files
    config['output_files'] = files
    cap.release()

    out = cv2.VideoWriter(f'{base_path}/frames.avi',cv2.VideoWriter_fourcc(*'MJPG'), config['video_freq'], (1024, 1024))
    for file in files:
        out.write(cv2.imread(file))
    out.release()

@hydra.main(config_name='configs/preprocess')
def main(config):
    actions = {
        'align': align_images,
        'interactive_mask': interactive_mask_images,
        'mask': mask_image,
        'blend': blend_images,
        'mash': mash_images,
        'remove_pixels': remove_pixels
    }
    datasets = {
        'CelebaHQ': CelebaHQDataset
    }
    if config['is_dataset']:
        # for now defaults to CelebaHQ
        dataset = datasets[config['dataset_type']](config['input_files'][0])
        config['input_files'] = random.sample(dataset.files, config['num_dataset'])
        print(f'Working with the following files: \n {config["input_files"]}')

    if config['preprocessing'][0] == 'collect_video_frames':
        collect_video_frames(config)
        config['preprocessing'] = config['preprocessing'][1:]

    if config['preprocessing'] and config['preprocessing'][0] == 'align':
        config['preprocessing'] = config['preprocessing'][1:]
        align_images(config)
    images = get_arrays(config)
    for action_name in config['preprocessing']:
        images = actions[action_name](config, images)
    out_index = 0
    curr_out_index = -1
    for index, image in enumerate(images):
        if curr_out_index == config['per_input'] - 1:
            curr_out_index = 0
            out_index += 1
        else:
            curr_out_index += 1
        base_name = config['input_files'][out_index].split('/')[-1].split('.')[0]
        file_ext = 'png'
        new_name = base_name + '_' + str(curr_out_index) + '.' + file_ext
        print(f'Saving output file: {new_name}')
        save_file(os.path.join(config['output_dir'], new_name), image)

if __name__ == '__main__':
    main()
