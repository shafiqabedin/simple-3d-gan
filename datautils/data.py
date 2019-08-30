import os
import random

import SimpleITK as sitk
import numpy as np

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

        print("- Data init finished")

    def generate_filenames(self):
        """
        Generates a list of files that end with .nii ot .gz
        :return: yields name and sitk image
        """
        for dir_path, dir_names, file_names in os.walk(os.path.join(data_root_path, 'train-images')):
            for file_name in file_names:
                if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                    image = sitk.Cast(sitk.ReadImage(os.path.join(dir_path, file_name)), sitk.sitkFloat32)
                    # print(image.GetPixelIDTypeAsString())
                    yield file_name, image

    def generate_array_from_nifti(self, files):
        """
        Takes in an iterable of filenames
        :param files: filenames
        :return: yields name and numpy array of image
        """
        for name, file in files:
            processed_image = sitk.GetArrayFromImage(file)[..., np.newaxis]
            yield name, processed_image

    def get_batch_volumes(self):
        # Generate list of filenames
        filenames = self.generate_filenames()

        # Generate numpy array of images
        images = self.generate_array_from_nifti(filenames)

        # Create a list for shuffle
        image_list = list(images)

        # Shuffle!
        random.shuffle(image_list)

        # Crete an empty volumes list
        volumes = []

        # Take the top - batch_size
        for idx, (name, image) in enumerate(image_list):
            volumes.append(image)
            # print("ID: " + str(idx) + " Name: " + str(name) + " Shape:" + str(image.shape))
            if idx >= self.batch_size - 1:
                break

        # Turn list to numpy array
        volumes = np.array(volumes)

        # Return the numpy array
        return volumes
