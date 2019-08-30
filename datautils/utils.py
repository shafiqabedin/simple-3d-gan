import os

import SimpleITK as sitk
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Qt4Agg')

# plt.close('all')
# plt.ion()


def save_volumes_as_images(volumes, n=1,
                           save_path='/gpfs/fs0/data/DeepLearning/sabedin/DeployedProjects/simple-3d-gan/data/tmp',
                           prefix=""):
    """
    Saves the given volume as image
    :param volumes: Collection of volumes
    :param n: number of volumes to process
    :return: None
    """
    # Set the prefix
    if prefix:
        prefix = prefix + "_"

    # Remove channel axis
    if volumes.ndim == 5:
        volumes = volumes[:, :, :, :, 0]

    for idx, volume in enumerate(volumes):
        image = sitk.GetImageFromArray(volume)
        sitk.WriteImage(image, os.path.join(save_path, prefix + str(idx) + ".nii"))

        if idx >= n - 1:
            break


def plot_volumes_as_images(volumes,
                           save_path='/gpfs/fs0/data/DeepLearning/sabedin/DeployedProjects/simple-3d-gan/data/tmp'):
    """
    Saves the given volume as image
    :param volumes: Collection of volumes
    :param n: number of volumes to process
    :return: None
    """

    fig = plt.figure(figsize=(10, 10))
    columns = 6
    rows = 6
    plt.gray()

    # Remove channel axis
    if volumes.ndim == 5:
        volumes = volumes[:, :, :, :, 0]

    for idx, volume in enumerate(volumes):
        image = volume[:, :, 32]
        # print(image.shape)
        fig.add_subplot(rows, columns, (idx + 1), title=str(idx + 1), xticks=[], yticks=[])
        plt.imshow(image)

    plt.show()
