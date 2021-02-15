import yaml
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def grab_images(category):
    paths = [x for x in glob.iglob(os.path.join(category + '*.png'))]
    paths.sort()
    images = np.stack([np.array(Image.open(x)) for x in paths])
    return paths, images

def calculate_error(x, y, config):
    # ignore masked positions
    mask = np.expand_dims((x.sum(axis=-1) != 0).astype(np.int64), -1)
    mean_image_error = ((x - y)**2 * mask).sum(axis=(1, 2, 3)) / (3 * mask.sum(axis=(1, 2, 3)))
    return mean_image_error.reshape(-1, len(config['percentages'])).mean(axis=0)


def plot_errors(errors, config, ylabel):
    for error in errors:
        plt.plot(config['percentages'], error, 'o-')
    plt.legend(labels)
    plt.xlabel('Percentage of observed pixels')
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        config = config['filter']

    # grab images
    inpainted_paths, inpainted_images = grab_images(config['inpainted'])
    labels = ['csgm_mse', 'csgm_lpips', 'csgm_mse_lpips', 'ilo']
    images = [grab_images(config[x])[1] for x in labels]

    # grab real images
    file_names_no_ext = [x.split('/')[-1].split('_')[0] for x in inpainted_paths]
    real_images = []
    for file_name in file_names_no_ext:
        img = np.array(Image.open(os.path.join(config['original'], file_name + '.' + config['image_type'])).convert('RGB'))
        real_images.append(img)
    real_images = np.stack(real_images)

    real_errors = [calculate_error(real_images, image, config) for image in images]
    obsv_errors = [calculate_error(inpainted_images, image, config) for image in images]

    plot_errors(real_errors, config, 'Real MSE')
    plot_errors(obsv_errors, config, 'Inpainted MSE')
