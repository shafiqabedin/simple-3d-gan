import os
import random

import numpy as np
from scipy.ndimage import zoom

data_root_path = "/gpfs/fs0/data/DeepLearning/sabedin/Data/sample_3d_gan"


class DataSet:
    """
    DataSet class
    """

    def __init__(self, batch_size):
        """
        Initialize the DataSet class
        Args:
            image_rows, image_cols: Training image size
            batch_size: Batch Size

        """

        print("- Data init started")
        self.batch_size = batch_size

        # Load data
        self.volumes = np.load(os.path.join(data_root_path, 'train-images-mnist/volumes.npy'))

        print("- Data init finished")

    def generate_filenames(self):
        """
        Generates a list of files that end with .nii ot .gz
        :return: yields name and sitk image
        """
        for dir_path, dir_names, file_names in os.walk(os.path.join(data_root_path, 'train-images-mnist')):
            for file_name in file_names:
                if file_name.endswith('.npz'):
                    image = np.load(os.path.join(dir_path, file_name))
                    # print(image.GetPixelIDTypeAsString())
                    yield file_name, image

    def generate_array_from_nifti(self, files):
        """
        Takes in an iterable of filenames
        :param files: filenames
        :return: yields name and numpy array of image
        """
        for name, file in files:
            processed_image = file["arr_0"]
            processed_image = np.mean(processed_image, axis=3)
            processed_image = zoom(processed_image, (4, 4, 4))

            processed_image = processed_image[..., np.newaxis]
            yield name, processed_image

    # def get_batch_volumes(self):
    #
    #     print ("MNIST DataSet")
    #     # Generate list of filenames
    #     filenames = self.generate_filenames()
    #
    #     # Generate numpy array of images
    #     images = self.generate_array_from_nifti(filenames)
    #
    #     # Create a list for shuffle
    #     image_list = list(images)
    #
    #     # Shuffle!
    #     random.shuffle(image_list)
    #
    #     # Crete an empty volumes list
    #     volumes = []
    #
    #     # Take the top - batch_size
    #     for idx, (name, image) in enumerate(image_list):
    #         volumes.append(image)
    #         # print("ID: " + str(idx) + " Name: " + str(name) + " Shape:" + str(image.shape))
    #         if idx >= self.batch_size - 1:
    #             break
    #
    #     # Turn list to numpy array
    #     volumes = np.array(volumes)
    #
    #     # print "Volume Shape: ", volumes.shape
    #
    #     # Return the numpy array
    #     return volumes

    def get_batch_volumes(self):

        # Turn list to numpy array
        list_of_random_volumes = np.asarray(random.sample(self.volumes, self.batch_size))
        return list_of_random_volumes
